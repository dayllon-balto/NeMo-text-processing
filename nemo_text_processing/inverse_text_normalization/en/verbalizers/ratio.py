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
    NEMO_DIGIT,
    GraphFst,
    delete_space,
)


class RatioFst(GraphFst):
    """
    Finite state transducer for verbalizing ratio, e.g.
        ratio { antecedent: "4" consequent: "2" } -> 4:2
        ratio { antecedent: "10" consequent: "1" } -> 10:1
    """

    def __init__(self):
        super().__init__(name="ratio", kind="verbalize")

        antecedent = (
            pynutil.delete("antecedent:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        consequent = (
            pynutil.delete("consequent:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )

        graph = antecedent + delete_space + pynutil.insert(":") + consequent

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
