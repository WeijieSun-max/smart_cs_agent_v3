from pathlib import Path

from infra.knowledge.document_loaders import load_10086_help, load_10086_qa


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_10086_qa_loader_indexes_each_question() -> None:
    records = [record for record in load_10086_qa(ROOT_DIR) if record.payload["source_file"] == "01_账户使用类.md"]

    assert len(records) == 27
    assert all(record.payload["first_level_index"] == "账户使用类" for record in records)
    assert all(record.payload["question"] for record in records)
    assert all(record.payload["answer"] for record in records)
    assert records[0].payload["question_index"] == 1
    assert "移动电话的漫游分为两种" in records[0].payload["answer"]


def test_10086_help_loader_indexes_each_document() -> None:
    records = load_10086_help(ROOT_DIR)
    files = list((ROOT_DIR / "data" / "10086_help").glob("*.md"))

    assert len(records) == len(files)
    registration = next(record for record in records if record.payload["source_file"] == "14_用户注册.md")
    assert registration.payload["first_level_index"] == "用户注册"
    assert registration.payload["title"] == "用户注册"
    assert registration.payload["source_type"] == "help"
    assert registration.payload["image_urls"]
    assert "邮箱帐号" in registration.content
