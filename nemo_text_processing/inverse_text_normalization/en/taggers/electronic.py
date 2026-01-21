# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path, get_various_formats
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MIN_POS_WEIGHT,
    NEMO_ALPHA,
    GraphFst,
    capitalized_input_graph,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import load_labels


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. c d f one at a b c dot e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="electronic", kind="classify")

        delete_extra_space = pynutil.delete(" ")

        num = pynini.string_file(get_abs_path("data/numbers/digit.tsv")) | pynini.string_file(
            get_abs_path("data/numbers/zero.tsv")
        )

        if input_case == INPUT_CASED:
            num = capitalized_input_graph(num)

        alpha_num = (NEMO_ALPHA | num).optimize()

        url_symbols = pynini.string_file(get_abs_path("data/electronic/url_symbols.tsv")).invert()
        accepted_username = alpha_num | url_symbols
        process_dot = pynini.cross("dot", ".")
        alternative_dot = (
            pynini.closure(delete_extra_space, 0, 1) + pynini.accep(".") + pynini.closure(delete_extra_space, 0, 1)
        )

        # Original pattern: single letters with spaces (e.g., "c d f one")
        username_spelled = (alpha_num + pynini.closure(delete_extra_space + accepted_username)) | pynutil.add_weight(
            pynini.closure(NEMO_ALPHA, 1), weight=0.0001
        )

        # New pattern: full words with dots (e.g., "david dot ayllon")
        # A word is a sequence of letters
        word = pynini.closure(NEMO_ALPHA, 1)
        # Username can be word, or word dot word, etc.
        username_words = word + pynini.closure(delete_extra_space + process_dot + delete_extra_space + word)

        # Combine both patterns
        username = username_spelled | username_words
        username = pynutil.insert("username: \"") + username + pynutil.insert("\"")
        single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num
        server = (
            single_alphanum
            | pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
            | pynini.closure(NEMO_ALPHA, 2)
        )

        if input_case == INPUT_CASED:
            domain = []
            # get domain formats
            for d in load_labels(get_abs_path("data/electronic/domain.tsv")):
                domain.extend(get_various_formats(d[0]))
            domain = pynini.string_map(domain).optimize()
        else:
            domain = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        domain = single_alphanum | domain | pynini.closure(NEMO_ALPHA, 2)
        domain_graph = (
            pynutil.insert("domain: \"")
            + server
            + ((delete_extra_space + process_dot + delete_extra_space) | alternative_dot)
            + domain
            + pynutil.insert("\"")
        )
        # Email pattern: username at domain
        # Give it lower weight to prefer email over URL protocol matching
        email_graph = username + delete_extra_space + pynutil.delete("at") + insert_space + delete_extra_space + domain_graph
        graph = pynutil.add_weight(email_graph, -0.5)

        ############# url ###
        if input_case == INPUT_CASED:
            www_graph = pynini.cross(pynini.union(*get_various_formats("www")), "www")

            protocol_scheme = pynini.cross(pynini.union(*get_various_formats("http")), "http") | pynini.cross(
                pynini.union(*get_various_formats("https")), "https"
            )
        else:
            www_graph = pynini.cross(pynini.union("w w w", "www"), "www")
            protocol_scheme = pynini.cross("h t t p", "http") | pynini.cross("h t t p s", "https")

        # "colon slash slash" -> "://"
        colon_slash_slash = pynini.cross(" colon slash slash ", "://") | pynini.cross(" colon slash slash", "://")

        # Full protocol with scheme: "h t t p s colon slash slash" -> "https://"
        protocol_with_scheme = protocol_scheme + colon_slash_slash

        # Process "slash" -> "/" for URL paths
        process_slash = pynini.cross("slash", "/")

        # URL path segments: "slash status" -> "/status", "slash v one" -> "/v1"
        # Each path segment must be explicitly preceded by "slash"
        # Path patterns:
        # 1. Single word: "status" -> "status"
        # 2. Letter(s) + number(s): "v one" -> "v1", "v two three" -> "v23"
        # 3. Single number: "one" -> "1"
        path_alpha = pynini.closure(NEMO_ALPHA, 1)
        path_num_seq = num + pynini.closure(delete_extra_space + num, 0, 3)
        # Alphanumeric: letters followed by numbers (merged without space)
        path_alphanum = path_alpha + delete_extra_space + path_num_seq
        # Path component: prefer alphanum combo, then single word, then single number
        path_component = (
            pynutil.add_weight(path_alphanum, -0.2)  # "v one" -> "v1"
            | pynutil.add_weight(path_num_seq, -0.1)  # "one two" -> "12"
            | path_alpha  # "status" -> "status"
        )
        path_segment = (
            delete_extra_space
            + process_slash
            + delete_extra_space
            + path_component
        )

        # Domain part: handles "example dot com" -> "example.com"
        # Also handles subdomains: "api dot example dot com" -> "api.example.com"
        domain_part = (
            pynini.closure(NEMO_ALPHA, 1)
            + pynini.closure(delete_extra_space + process_dot + delete_extra_space + pynini.closure(NEMO_ALPHA, 1), 1)
        )

        # .com, ending patterns (original)
        ending = (
            delete_extra_space
            + url_symbols
            + delete_extra_space
            + (
                domain
                | pynini.closure(
                    accepted_username + delete_extra_space,
                )
                + accepted_username
            )
        )

        protocol_default = (
            (
                (pynini.closure(delete_extra_space + accepted_username, 1) | server)
                | pynutil.add_weight(pynini.closure(NEMO_ALPHA, 1), weight=0.0001)
            )
            + pynini.closure(ending, 1)
        ).optimize()

        # URL with scheme but no www: "h t t p s colon slash slash example dot com" -> "https://example.com"
        url_with_scheme_no_www = (
            protocol_with_scheme
            + delete_extra_space
            + domain_part
            + pynini.closure(path_segment)
        )

        # URL with www: "w w w dot example dot com" or "h t t p s colon slash slash w w w dot example dot com"
        url_with_www = (
            pynini.closure(protocol_with_scheme + delete_extra_space, 0, 1)
            + www_graph
            + delete_extra_space
            + process_dot
            + protocol_default
        )

        if input_case == INPUT_CASED:
            url_with_www |= (
                pynini.closure(protocol_with_scheme + delete_extra_space, 0, 1)
                + www_graph
                + alternative_dot
                + protocol_default
            ).optimize()

        # URL without scheme or www (original pattern)
        url_no_scheme = pynini.closure(www_graph + delete_extra_space + process_dot, 0, 1) + protocol_default

        # Combine all URL patterns (prefer more specific patterns)
        protocol = (
            pynutil.add_weight(url_with_scheme_no_www, -0.5)  # Prefer full URL with scheme
            | pynutil.add_weight(url_with_www, -0.3)  # Then URL with www
            | url_no_scheme  # Fallback to original pattern
        )

        protocol = pynutil.insert("protocol: \"") + protocol.optimize() + pynutil.insert("\"")
        graph |= protocol

        if input_case == INPUT_CASED:
            graph = capitalized_input_graph(graph, capitalized_graph_weight=MIN_POS_WEIGHT)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
