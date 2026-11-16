"""Splitting generated prompts into a question and four options, and reading the key."""
from check2.data.expansion.parse import parse_key, split_prompt


def test_options_in_blocks():
    p = split_prompt("What is 2 + 2?\n\n3\n\n4\n\n5\n\n22")
    assert p.question == "What is 2 + 2?"
    assert p.choices == ("3", "4", "5", "22")


def test_options_on_single_lines():
    p = split_prompt("What is 2 + 2?\n3\n4\n5\n22")
    assert p.question == "What is 2 + 2?"
    assert p.choices == ("3", "4", "5", "22")


def test_question_with_a_blank_line_and_options_on_single_lines():
    p = split_prompt("What does this print?\n\nprint(1 + 1)\n\n1\n2\n11\nError")
    assert p.question == "What does this print?\n\nprint(1 + 1)"
    assert p.choices == ("1", "2", "11", "Error")


def test_question_lines_before_options_in_the_last_block():
    p = split_prompt("Consider:\nSET x = 1\nWhat is x?\n1\n2\n3\n4")
    assert p.question == "Consider:\nSET x = 1\nWhat is x?"
    assert p.choices == ("1", "2", "3", "4")


def test_multi_line_question_with_options_in_blocks():
    p = split_prompt(" Premises:\nA > B\nB > C\n\nA > C\n\nC > A\n\nB > A\n\nnone")
    assert p.question == "Premises:\nA > B\nB > C"
    assert p.choices == ("A > C", "C > A", "B > A", "none")


def test_options_in_blocks_with_a_leading_space():
    p = split_prompt("Characteristic of Z_p?\n\n 0\n\n 1\n\n p\n\n p^2")
    assert p.choices == ("0", "1", "p", "p^2")


def test_labelled_options():
    p = split_prompt("Which follows?\n\nA. p\nB. q\nC. r\nD. s")
    assert p.question == "Which follows?"
    assert p.choices == ("p", "q", "r", "s")


def test_extra_blank_lines_are_one_separator():
    p = split_prompt("Efficiency?\n\n\n25%\n\n40%\n\n75%\n\n50%")
    assert p.choices == ("25%", "40%", "75%", "50%")


def test_options_run_together_on_one_line_do_not_split():
    assert split_prompt("Which is right? Prices rise Prices fall Prices hold Prices vanish") is None


def test_repeated_options_do_not_split():
    assert split_prompt("What prints?\n\n[1]\n[2]\n[1]\n[1, 2]") is None


def test_options_equal_once_trimmed_do_not_split():
    assert split_prompt("Pick one.\n\nAlpha\n\n Alpha \n\nBeta\n\nGamma") is None


def test_options_that_differ_only_in_spacing_are_distinct():
    p = split_prompt("What does print(3 * 'A') show?\n\n'AAA'\n'A A A'\n'3A'\nError")
    assert p.choices == ("'AAA'", "'A A A'", "'3A'", "Error")


def test_indented_line_is_not_an_option():
    assert split_prompt("Which opens a file?\n\nopen(f)\nwith open(f) as g:\n    g.read()\nfile_open(f)\nread(f)") is None


def test_no_question_does_not_split():
    assert split_prompt("true\nfalse\nNone\nError") is None


def test_too_few_options_do_not_split():
    assert split_prompt("Pick one.\n\nn\nn-1\n\nn+1\n\n2n") is None


def test_several_questions_in_one_prompt_do_not_split():
    first = "First question?\n\na\n\nb\n\nc\n\nd"
    second = "Second question?\n\ne\n\nf\n\ng\n\nh"
    assert split_prompt(first + "\n\n\n" + second) is None


def test_key_is_one_letter():
    assert parse_key("B") == "B"
    assert parse_key(" C\n") == "C"
    assert parse_key("E") is None
    assert parse_key("AB") is None
    assert parse_key("") is None
    assert parse_key("b") is None
