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
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class ScoreFst(GraphFst):
    """
    Finite state transducer for classifying scores
        e.g. seven six -> score { home: "7" away: "6" }
        e.g. nine five -> score { home: "9" away: "5" }
        e.g. twenty one six -> score { home: "21" away: "6" }

    To avoid ambiguity with years and large numbers, only matches scores where
    at least one number is a single digit (0-9). This covers most sports scores.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="score", kind="classify")

        # Load number graphs for score-like numbers
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))

        # Handle casing
        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA).optimize()

        # Single digits (0-9) - these are "simple" numbers
        single_digit = pynini.compose(casing_graph, graph_zero | graph_digit).optimize()

        # Two-digit numbers (10-99)
        # Teens: 10-19 (single word, less ambiguous)
        teens = pynini.compose(casing_graph, graph_teen).optimize()
        # Ties with digit: 20-29, 30-39, etc. (compound - "twenty one")
        ties = pynini.compose(casing_graph, graph_ties).optimize()
        ties_with_digit = ties + delete_space + pynini.compose(casing_graph, graph_digit).optimize()
        ties_alone = ties + pynutil.insert("0")

        # Any score number (0-99)
        any_score_number = single_digit | teens | pynutil.add_weight(ties_with_digit, -0.1) | ties_alone

        # To avoid ambiguity, require at least one single digit in the score
        # This prevents "twenty one fourteen" from matching (ambiguous with 2114)
        # But allows "seven six", "twenty one six", "six twenty one", etc.
        graph = (
            # Single digit vs any number
            (
                pynutil.insert("home: \"")
                + single_digit
                + pynutil.insert("\"")
                + delete_extra_space
                + pynutil.insert("away: \"")
                + any_score_number
                + pynutil.insert("\"")
            )
            |
            # Any number vs single digit (when first is not single digit)
            (
                pynutil.insert("home: \"")
                + (teens | ties_with_digit | ties_alone)
                + pynutil.insert("\"")
                + delete_extra_space
                + pynutil.insert("away: \"")
                + single_digit
                + pynutil.insert("\"")
            )
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
