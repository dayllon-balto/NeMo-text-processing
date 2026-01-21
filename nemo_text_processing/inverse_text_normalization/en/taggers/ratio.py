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

from nemo_text_processing.text_normalization.en.graph_utils import (
    INPUT_CASED,
    INPUT_LOWER_CASED,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
)


class RatioFst(GraphFst):
    """
    Finite state transducer for classifying ratios
        e.g. four to one -> ratio { antecedent: "4" consequent: "1" }
        e.g. ten to one -> ratio { antecedent: "10" consequent: "1" }
        e.g. three to two -> ratio { antecedent: "3" consequent: "2" }

    Only matches patterns that are clearly ratios:
    - "X to one" (most common ratio pattern)
    - Small number ratios like "three to two", "four to three"

    Args:
        cardinal: CardinalFst
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, cardinal: GraphFst, input_case: str = INPUT_LOWER_CASED):
        super().__init__(name="ratio", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        # Common ratio denominators: one, two, three, four, five
        # This prevents matching ranges like "from one to ten"
        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA).optimize()
        ratio_denominators = pynini.string_map([
            ("one", "1"),
            ("two", "2"),
            ("three", "3"),
            ("four", "4"),
            ("five", "5"),
        ])
        ratio_denominators = pynini.compose(casing_graph, ratio_denominators).optimize()

        # "to" keyword between the two numbers
        to_keyword = pynutil.delete("to")
        if input_case == INPUT_CASED:
            to_keyword = pynutil.delete(pynini.union("to", "To", "TO"))

        # Build the ratio graph - only match when consequent is a common ratio denominator
        # This prevents matching ranges like "from one to ten"
        graph = (
            pynutil.insert("antecedent: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + to_keyword
            + delete_extra_space
            + pynutil.insert("consequent: \"")
            + ratio_denominators
            + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
