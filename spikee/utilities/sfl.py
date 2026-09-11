"""Parser and evaluator for the Spikee Filter Language (SFL)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


class SFLSyntaxError(ValueError):
    """Raised when an SFL expression cannot be parsed."""

    def __init__(self, query: str, position: int, message: str):
        self.query = query
        self.position = position
        super().__init__(f"SFL syntax error at character {position + 1}: {message}")


@dataclass(frozen=True)
class _Token:
    kind: str
    value: Any
    position: int


@dataclass(frozen=True)
class Expression:
    """Base class for parsed SFL expressions."""


@dataclass(frozen=True)
class BooleanExpression(Expression):
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True)
class NotExpression(Expression):
    expression: Expression


@dataclass(frozen=True)
class FullTextExpression(Expression):
    value: str


@dataclass(frozen=True)
class ExistsExpression(Expression):
    field: str


@dataclass(frozen=True)
class ComparisonExpression(Expression):
    field: str
    operator: str
    value: Any


_KEYWORDS = {"AND", "OR", "NOT", "CONTAINS", "HAS", "LIKE", "EXISTS"}
_MISSING = object()


def parse_sfl(query: str) -> Expression:
    """Parse an SFL query into an expression tree."""
    return _Parser(query).parse()


def matches_sfl(entry: dict[str, Any], expression: Expression) -> bool:
    """Return whether a result entry matches a parsed SFL expression."""
    match expression:
        case BooleanExpression("AND", left, right):
            return matches_sfl(entry, left) and matches_sfl(entry, right)
        case BooleanExpression("OR", left, right):
            return matches_sfl(entry, left) or matches_sfl(entry, right)
        case NotExpression(inner):
            return not matches_sfl(entry, inner)
        case FullTextExpression(value):
            return any(_contains(item, value) for item in _entry_values(entry))
        case ExistsExpression(field):
            value = _get_field(entry, field)
            return value is not _MISSING and value is not None
        case ComparisonExpression(field, operator, value):
            actual = _get_field(entry, field)
            return _compare(actual, operator, value)
        case _:
            raise TypeError(f"Unsupported SFL expression: {expression!r}")


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if actual is _MISSING:
        return False
    if operator == "=":
        if isinstance(expected, str) and isinstance(actual, str):
            return actual.casefold() == expected.casefold()
        return actual == expected
    if operator == "!=":
        return not _compare(actual, "=", expected)
    if operator == "LIKE":
        pattern = re.escape(str(expected)).replace("%", ".*").replace("_", ".")
        return re.fullmatch(pattern, str(actual), flags=re.IGNORECASE) is not None
    if operator == "HAS":
        if not isinstance(actual, (list, tuple, set)):
            return False
        return any(_equals(item, expected) for item in actual)
    if operator in {"<", "<=", ">", ">="}:
        if not _is_number(actual) or not _is_number(expected):
            return False
        return {
            "<": actual < expected,
            "<=": actual <= expected,
            ">": actual > expected,
            ">=": actual >= expected,
        }[operator]
    raise ValueError(f"Unsupported SFL comparison operator: {operator}")


def _equals(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.casefold() == expected.casefold()
    return actual == expected


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _contains(value: Any, expected: str) -> bool:
    return expected.casefold() in str(value).casefold()


def _entry_values(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _entry_values(item)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _entry_values(item)
    else:
        yield value


def _get_field(entry: dict[str, Any], field: str) -> Any:
    value: Any = entry
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


class _Parser:
    def __init__(self, query: str):
        self.query = query
        self.tokens = list(_tokenize(query))
        self.index = 0

    def parse(self) -> Expression:
        if self._current().kind == "EOF":
            self._error(self._current(), "expected an expression")
        expression = self._parse_or()
        if self._current().kind != "EOF":
            self._error(self._current(), "expected AND, OR, or end of query")
        return expression

    def _parse_or(self) -> Expression:
        expression = self._parse_and()
        while self._match_keyword("OR"):
            expression = BooleanExpression("OR", expression, self._parse_and())
        return expression

    def _parse_and(self) -> Expression:
        expression = self._parse_not()
        while self._match_keyword("AND"):
            expression = BooleanExpression("AND", expression, self._parse_not())
        return expression

    def _parse_not(self) -> Expression:
        if self._match_keyword("NOT"):
            return NotExpression(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        if self._match("LPAREN"):
            expression = self._parse_or()
            self._expect("RPAREN", "expected ')' to close expression")
            return expression
        if self._current().kind == "STRING":
            return FullTextExpression(self._advance().value)
        return self._parse_condition()

    def _parse_condition(self) -> Expression:
        field = self._expect("IDENT", "expected a field name or quoted text").value
        if self._match_keyword("EXISTS"):
            return ExistsExpression(field)
        token = self._current()
        if (
            token.kind == "OPERATOR"
            or token.kind == "KEYWORD"
            and token.value in {"HAS", "LIKE"}
        ):
            operator = self._advance().value
        elif token.kind == "KEYWORD" and token.value == "CONTAINS":
            self._error(token, "CONTAINS is not supported; use LIKE with '%' wildcards")
        else:
            return ExistsExpression(field)
        return ComparisonExpression(field, operator, self._parse_value())

    def _parse_value(self) -> Any:
        token = self._current()
        if token.kind in {"STRING", "IDENT"}:
            return self._advance().value
        self._error(token, "expected a comparison value")

    def _current(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self._current()
        self.index += 1
        return token

    def _match(self, kind: str) -> bool:
        if self._current().kind == kind:
            self._advance()
            return True
        return False

    def _match_keyword(self, keyword: str) -> bool:
        token = self._current()
        if token.kind == "KEYWORD" and token.value == keyword:
            self._advance()
            return True
        return False

    def _expect(self, kind: str, message: str) -> _Token:
        if self._current().kind == kind:
            return self._advance()
        self._error(self._current(), message)

    def _error(self, token: _Token, message: str) -> None:
        raise SFLSyntaxError(self.query, token.position, message)


def _tokenize(query: str) -> Iterator[_Token]:
    position = 0
    while position < len(query):
        character = query[position]
        if character.isspace():
            position += 1
        elif character in "()":
            yield _Token(
                "LPAREN" if character == "(" else "RPAREN", character, position
            )
            position += 1
        elif query.startswith("!=", position):
            yield _Token("OPERATOR", "!=", position)
            position += 2
        elif query.startswith((">=", "<="), position):
            yield _Token("OPERATOR", query[position : position + 2], position)
            position += 2
        elif character in "=><":
            yield _Token("OPERATOR", character, position)
            position += 1
        elif character == '"':
            yield _read_string(query, position)
            position = _read_string_end(query, position)
        else:
            match = re.match(
                r"[A-Za-z_][A-Za-z0-9_.-]*|-?\d+(?:\.\d+)?", query[position:]
            )
            if not match:
                raise SFLSyntaxError(
                    query, position, f"unexpected character {character!r}"
                )
            value = match.group(0)
            upper_value = value.upper()
            if upper_value in _KEYWORDS:
                yield _Token("KEYWORD", upper_value, position)
            elif value.casefold() == "true":
                yield _Token("IDENT", True, position)
            elif value.casefold() == "false":
                yield _Token("IDENT", False, position)
            elif value.casefold() == "null":
                yield _Token("IDENT", None, position)
            elif re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                yield _Token(
                    "IDENT", float(value) if "." in value else int(value), position
                )
            else:
                yield _Token("IDENT", value, position)
            position += len(value)
    yield _Token("EOF", None, position)


def _read_string(query: str, start: int) -> _Token:
    end = _read_string_end(query, start)
    try:
        value = bytes(query[start + 1 : end - 1], "utf-8").decode("unicode_escape")
    except UnicodeDecodeError as error:
        raise SFLSyntaxError(query, start, "invalid string escape") from error
    return _Token("STRING", value, start)


def _read_string_end(query: str, start: int) -> int:
    position = start + 1
    characters: list[str] = []
    while position < len(query):
        character = query[position]
        if character == '"':
            return position + 1
        if character == "\\":
            position += 1
            if position == len(query):
                break
            character = query[position]
        characters.append(character)
        position += 1
    raise SFLSyntaxError(query, start, "unterminated quoted string")
