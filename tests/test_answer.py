"""Answering contract: no context means no answer, and citations come from
retrieval rather than from the model's imagination."""

from src.rag.answer import ABSTENTION_TEXT, Answerer, EchoAnswerer, build_prompt


def test_abstains_on_empty_contexts():
    result = EchoAnswerer().answer("what is the per diem cap?", [])
    assert result["abstained"] is True
    assert result["answer"] == ABSTENTION_TEXT
    assert result["cited_sources"] == []


def test_does_not_abstain_when_contexts_exist():
    result = EchoAnswerer().answer("what is the per diem cap?", ["expense-policy"])
    assert result["abstained"] is False
    assert result["cited_sources"] == ["expense-policy"]


def test_cited_sources_mirror_the_retrieved_contexts():
    contexts = ["expense-policy", "travel-policy", "security-policy"]
    assert EchoAnswerer().answer("q?", contexts)["cited_sources"] == contexts


def test_result_has_the_documented_keys():
    assert set(EchoAnswerer().answer("q?", ["doc"])) == {
        "answer",
        "cited_sources",
        "abstained",
    }


def test_answer_is_deterministic():
    answerer = EchoAnswerer()
    assert answerer.answer("q?", ["doc"]) == answerer.answer("q?", ["doc"])


def test_echo_answerer_is_an_answerer():
    assert isinstance(EchoAnswerer(), Answerer)


def test_cannot_instantiate_the_abstract_base():
    try:
        Answerer()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Answerer should not be instantiable")


def test_build_prompt_includes_question_and_every_context():
    prompt = build_prompt("per diem cap?", ["doc one", "doc two"])
    assert "per diem cap?" in prompt
    assert "[1] doc one" in prompt
    assert "[2] doc two" in prompt
    assert "only the numbered context" in prompt
