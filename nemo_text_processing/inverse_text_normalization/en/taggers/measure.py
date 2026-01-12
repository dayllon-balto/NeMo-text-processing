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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    MINUS,
    NEMO_ALPHA,
    NEMO_SIGMA,
    NEMO_SPACE,
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    get_singulars,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. minus twelve kilograms -> measure { negative: "true" cardinal { integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        # accept capital letters in units
        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA).optimize()

        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_unit_singular = pynini.invert(graph_unit)  # singular -> abbr
        graph_unit_singular = pynini.compose(casing_graph, graph_unit_singular).optimize()

        graph_unit_plural = get_singulars(graph_unit_singular).optimize()  # plural -> abbr
        graph_unit_plural = pynini.compose(casing_graph, graph_unit_plural).optimize()

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross(MINUS, "\"true\"") + delete_extra_space,
            0,
            1,
        )

        unit_singular = convert_space(graph_unit_singular)
        unit_plural = convert_space(graph_unit_plural)
        unit_misc = pynutil.insert("/") + pynutil.delete("per") + delete_space + convert_space(graph_unit_singular)

        one_graph = pynini.accep("one").optimize()
        if input_case == INPUT_CASED:
            one_graph |= pynini.accep("One").optimize()

        unit_singular = (
            pynutil.insert("units: \"")
            + (unit_singular | unit_misc | pynutil.add_weight(unit_singular + delete_space + unit_misc, 0.01))
            + pynutil.insert("\"")
        )
        unit_plural = (
            pynutil.insert("units: \"")
            + (unit_plural | unit_misc | pynutil.add_weight(unit_plural + delete_space + unit_misc, 0.01))
            + pynutil.insert("\"")
        )

        # Let singular apply to values > 1 as they could be part of an adjective phrase (e.g. 14 foot tall building)
        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + (unit_plural | unit_singular)
        )
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - one_graph) @ cardinal_graph)
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + (unit_plural | unit_singular)
        )
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross(one_graph, "1")
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_singular
        )
        # Address graph for street addresses
        # Give it a higher weight so it's less preferred than other patterns
        address_graph = self.get_address_graph(cardinal, input_case)
        address = (
            pynutil.insert('units: "address" cardinal { integer: "')
            + address_graph
            + pynutil.insert('" } preserve_order: true')
        )
        address = pynutil.add_weight(address, 1.0)

        final_graph = subgraph_decimal | subgraph_cardinal | address
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_address_graph(self, cardinal: GraphFst, input_case: str):
        """
        Finite state transducer for classifying addresses.
            e.g. one twenty three main street -> measure { units: "address" cardinal { integer: "123 Main St" } preserve_order: true }
            e.g. forty five elm avenue -> measure { units: "address" cardinal { integer: "45 Elm Ave" } preserve_order: true }

        Address numbers are typically spoken as:
            - "123" -> "one twenty three" (single digit + two digits)
            - "1234" -> "twelve thirty four" (two digits + two digits)
            - "45" -> "forty five" (two digits)
            - "5" -> "five" (single digit)

        Args:
            cardinal: CardinalFst
            input_case: accepting either "lower_cased" or "cased" input.
        """
        # Load number word graphs
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        # Handle casing for number words
        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA).optimize()
        graph_digit_cased = pynini.compose(casing_graph, graph_digit).optimize()
        graph_ties_cased = pynini.compose(casing_graph, graph_ties).optimize()
        graph_teen_cased = pynini.compose(casing_graph, graph_teen).optimize()

        # Two-digit numbers: teens (11-19) or ties+digit (20-99)
        # "twenty" alone -> "20", "twenty three" -> "23"
        graph_two_digit_with_ones = graph_ties_cased + delete_space + graph_digit_cased
        graph_two_digit_round = graph_ties_cased + pynutil.insert("0")
        graph_two_digit = (
            graph_teen_cased
            | pynutil.add_weight(graph_two_digit_with_ones, -0.1)  # Prefer "forty five" over "forty"
            | graph_two_digit_round
        )

        # Address number patterns:
        # 2-digit: "forty five" -> "45" or "fifteen" -> "15"
        # 3-digit: "one twenty three" -> "123" (single + two)
        # 4-digit: "twelve thirty four" -> "1234" (two + two)
        # 5-digit: "nineteen one nine five" -> "19195" (teen + digits)
        address_num_2digit = graph_two_digit
        address_num_3digit = graph_digit_cased + delete_space + graph_two_digit
        address_num_4digit = graph_two_digit + delete_space + graph_two_digit

        # For longer addresses (5+ digits), support digit-by-digit patterns
        # e.g., "nineteen one nine five" -> "19195" (teen + digit + digit + digit)
        # e.g., "one two three four five" -> "12345" (digit + digit + digit + digit + digit)
        digit_seq = graph_digit_cased + pynini.closure(delete_space + graph_digit_cased, 1, 4)
        address_num_5digit_teen = graph_teen_cased + delete_space + digit_seq
        address_num_5digit_digits = graph_digit_cased + delete_space + digit_seq

        # Combine address number patterns (prefer longer matches with weights)
        address_num = (
            pynutil.add_weight(address_num_5digit_teen, -0.4)
            | pynutil.add_weight(address_num_5digit_digits, -0.3)
            | pynutil.add_weight(address_num_4digit, -0.2)
            | pynutil.add_weight(address_num_3digit, -0.1)
            | address_num_2digit
        )

        # Load address word mappings (street -> St, avenue -> Ave, etc.)
        # Prefer longer matches (e.g., "highway" over "way") by weighting them
        address_words_graph = pynini.string_file(get_abs_path("data/address/address_word.tsv"))
        address_words_graph = pynini.compose(casing_graph, address_words_graph).optimize()
        # Handle compound words that might be split (highway vs high+way)
        # Give extra weight to longer address words
        long_address_words = (
            pynini.cross("highway", "Hwy")
            | pynini.cross("freeway", "Fwy")
            | pynini.cross("expressway", "Expy")
            | pynini.cross("boulevard", "Blvd")
            | pynini.cross("junction", "Jct")
        )
        long_address_words = pynini.compose(casing_graph, long_address_words).optimize()
        address_words_graph = pynutil.add_weight(long_address_words, -0.5) | address_words_graph

        # Street name must not be a number word
        # We list common short street names (2-4 chars) that are NOT number words
        short_street_names = pynini.union(
            # 2-3 char names
            "oak", "elm", "bay", "lee", "ash", "fir", "ivy", "jay", "key", "rio",
            "ada", "ava", "eve", "ida", "ora", "roy", "ray", "rex", "max", "sam",
            # 4 char names (excluding number words: one, two, four, five, nine, zero)
            "main", "pine", "park", "lake", "hill", "view", "palm", "rose", "king",
            "west", "east", "north", "south", "high", "mill", "wood", "glen", "dale",
            "ford", "port", "vale", "vine", "wolf", "bear", "deer", "hawk", "swan",
        )
        short_street_names = pynini.compose(casing_graph, short_street_names).optimize()

        # For 5+ char names, allow any alphabetic word
        # This may include some number words (seven, eight, three) but they're rarely street names
        long_street_name = NEMO_ALPHA + NEMO_ALPHA + NEMO_ALPHA + NEMO_ALPHA + pynini.closure(NEMO_ALPHA, 1)

        street_name = short_street_names | long_street_name

        # Build the address graph:
        # [address number] + [space] + [street name] + [space] + [address word]
        address = (
            address_num
            + pynutil.insert(" ")
            + delete_space
            + street_name
            + delete_space
            + pynutil.insert(" ")
            + address_words_graph
        )

        return address.optimize()
