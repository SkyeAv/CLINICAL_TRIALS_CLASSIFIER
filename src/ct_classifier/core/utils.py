from pydantic import ValidationError, BaseModel
from ruamel.yaml.error import YAMLError
from typing import TypeVar, Any
from ruamel.yaml import YAML
from pathlib import Path
import shutil
import lzma

yaml = YAML()


def load_yaml(file_path: Path) -> Any:
    posix_file_path: str = file_path.as_posix()
    try:
        with file_path.open("r") as f:
            return yaml.load(f)
    except FileNotFoundError:
        msg: str = f"CODE:1 | {posix_file_path} not found"
        raise RuntimeError(msg)
    except PermissionError:
        msg = f"CODE:2 | Permission denied: {posix_file_path}"
        raise RuntimeError(msg)
    except YAMLError as e:
        err: str = str(e)
        msg = f"CODE:3 | YAML parsing error in {posix_file_path} | {err}"
        raise RuntimeError(msg)


PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


def load_model(python_object: Any, pydantic_model: PydanticModel) -> dict[str, Any]:
    try:
        return pydantic_model.model_validate(python_object).model_dump()
    except ValidationError as e:
        val_err: str = str(e)
        msg: str = f"CODE:4 | Failed to parse to {pydantic_model.__name__} | {val_err}"  # type: ignore
        raise RuntimeError(msg)


def load_labels(label_file: Path) -> list[str]:
    return label_file.read_text(encoding="utf-8").splitlines()


def xz_backup(db_p: Path, fmt: str = r"%Y%m%d") -> None:
    timestamp: str = datetime.now().strftime(fmt)
    xz_p = db_p.with_name(db_p.stem + timestamp + db_p.suffix + ".xz")
    with db_p.open("rb") as f_in, lzma.open(xz_p, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return None
