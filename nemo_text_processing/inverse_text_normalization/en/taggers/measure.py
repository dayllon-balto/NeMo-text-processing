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
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        # Handle casing for number words
        casing_graph = pynini.closure(TO_LOWER | NEMO_SIGMA).optimize()
        graph_digit_cased = pynini.compose(casing_graph, graph_digit).optimize()
        graph_zero_cased = pynini.compose(casing_graph, graph_zero).optimize()
        # Combined digit+zero for sequences that can include zero/oh (e.g., "five oh one" -> "501")
        graph_digit_with_zero = pynini.compose(casing_graph, graph_digit | graph_zero).optimize()
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

        # "hundred" keyword for 3-digit numbers
        hundred = pynini.cross("hundred", "")
        if input_case == INPUT_CASED:
            hundred |= pynini.cross("Hundred", "")

        # Address number patterns:
        # 2-digit: "forty five" -> "45" or "fifteen" -> "15"
        # 3-digit: "one twenty three" -> "123" (single + two)
        # 3-digit with hundred: "four hundred twenty four" -> "424"
        # 4-digit: "twelve thirty four" -> "1234" (two + two)
        # 5-digit: "nineteen one nine five" -> "19195" (teen + digits)
        address_num_2digit = graph_two_digit

        # 3-digit patterns
        address_num_3digit = graph_digit_cased + delete_space + graph_two_digit
        # "X hundred" -> "X00" (e.g., "four hundred" -> "400")
        address_num_hundred_round = graph_digit_cased + delete_space + hundred + pynutil.insert("00")
        # "X hundred Y" -> "X0Y" (e.g., "four hundred five" -> "405")
        address_num_hundred_single = (
            graph_digit_cased + delete_space + hundred + delete_space + pynutil.insert("0") + graph_digit_cased
        )
        # "X hundred YZ" -> "XYZ" (e.g., "four hundred twenty four" -> "424")
        address_num_hundred_two = (
            graph_digit_cased + delete_space + hundred + delete_space + graph_two_digit
        )

        address_num_4digit = graph_two_digit + delete_space + graph_two_digit

        # For longer addresses (5+ digits), support digit-by-digit patterns
        # e.g., "nineteen one nine five" -> "19195" (teen + digit + digit + digit)
        # e.g., "one two three four five" -> "12345" (digit + digit + digit + digit + digit)
        # e.g., "five oh one" -> "501" (digit + zero/oh + digit)
        # Use graph_digit_with_zero to support "oh" as zero in sequences
        digit_seq = graph_digit_with_zero + pynini.closure(delete_space + graph_digit_with_zero, 1, 4)
        address_num_5digit_teen = graph_teen_cased + delete_space + digit_seq
        address_num_5digit_digits = graph_digit_cased + delete_space + digit_seq

        # Combine address number patterns (prefer longer matches with weights)
        address_num = (
            pynutil.add_weight(address_num_5digit_teen, -0.5)
            | pynutil.add_weight(address_num_5digit_digits, -0.4)
            | pynutil.add_weight(address_num_hundred_two, -0.35)  # Prefer "four hundred twenty four" -> 424
            | pynutil.add_weight(address_num_hundred_single, -0.3)  # Prefer "four hundred five" -> 405
            | pynutil.add_weight(address_num_hundred_round, -0.25)  # Prefer "four hundred" -> 400
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

        # Unit/suite designations (apartment, suite, unit, etc.)
        # Maps spoken form to abbreviation
        unit_words = (
            pynini.cross("apartment", "Apt")
            | pynini.cross("suite", "Ste")
            | pynini.cross("unit", "Unit")
            | pynini.cross("building", "Bldg")
            | pynini.cross("floor", "Fl")
            | pynini.cross("room", "Rm")
        )
        unit_words = pynini.compose(casing_graph, unit_words).optimize()

        # Unit number can be:
        # - Just a number: "apartment ten" -> "Apt 10"
        # - Letter + number: "apartment a ten" -> "Apt A10"
        # - Just a letter: "apartment b" -> "Apt B"
        # - Number + letter: "suite two a" -> "Ste 2A" (less common but possible)

        # Single letter (a-z) - uppercase in output
        single_letter = pynini.string_map([
            ("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "E"), ("f", "F"),
            ("g", "G"), ("h", "H"), ("i", "I"), ("j", "J"), ("k", "K"), ("l", "L"),
            ("m", "M"), ("n", "N"), ("o", "O"), ("p", "P"), ("q", "Q"), ("r", "R"),
            ("s", "S"), ("t", "T"), ("u", "U"), ("v", "V"), ("w", "W"), ("x", "X"),
            ("y", "Y"), ("z", "Z"),
        ])
        single_letter = pynini.compose(casing_graph, single_letter).optimize()

        # Unit number patterns (using the same number graphs as address numbers)
        # Two-digit: "ten", "twenty five"
        unit_num_2digit = graph_two_digit
        # Single digit for units: "five" -> "5"
        unit_num_1digit = graph_digit_cased

        unit_number = (
            pynutil.add_weight(unit_num_2digit, -0.1)  # Prefer two-digit
            | unit_num_1digit
        )

        # Combined unit designations:
        # "a ten" -> "A10" (letter followed by number, no space between)
        # "ten" -> "10" (just number)
        # "b" -> "B" (just letter)
        # "ten a" -> "10A" (number followed by letter)
        unit_letter_number = single_letter + delete_space + unit_number  # "a ten" -> "A10"
        unit_number_letter = unit_number + delete_space + single_letter  # "ten a" -> "10A"
        unit_designation = (
            pynutil.add_weight(unit_letter_number, -0.2)  # Prefer "a ten" -> "A10"
            | pynutil.add_weight(unit_number_letter, -0.1)  # Then "ten a" -> "10A"
            | unit_number  # Then just number
            | single_letter  # Then just letter
        )

        # Full unit suffix: "apartment a ten" -> " Apt A10"
        unit_suffix = (
            delete_space
            + pynutil.insert(" ")
            + unit_words
            + delete_space
            + pynutil.insert(" ")
            + unit_designation
        )

        # Build the address graph:
        # [address number] + [space] + [street name] + [space] + [address word] + [optional unit]
        address = (
            address_num
            + pynutil.insert(" ")
            + delete_space
            + street_name
            + delete_space
            + pynutil.insert(" ")
            + address_words_graph
            + pynini.closure(unit_suffix, 0, 1)
        )

        return address.optimize()
