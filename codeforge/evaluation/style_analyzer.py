"""Coding Style Fingerprint Engine.

Analyzes Python code style via AST and compares against known patterns.
See codeforge/evaluation/style_analyzer.py for the full implementation.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StyleFingerprint:
    """Quantified coding style profile."""
    snake_case_ratio: float = 0.0
    avg_name_length: float = 0.0
    descriptive_names_ratio: float = 0.0
    avg_function_length: float = 0.0
    docstring_coverage: float = 0.0
    type_hint_coverage: float = 0.0
    comment_density: float = 0.0
    list_comp_frequency: float = 0.0
    f_string_ratio: float = 0.0
    try_except_frequency: float = 0.0


def analyze_code_style(code: str) -> StyleFingerprint:
    """Analyze a code snippet and extract its style fingerprint."""
    fp = StyleFingerprint()
    lines = code.split("\n")
    total_lines = len([l for l in lines if l.strip()])
    if total_lines == 0:
        return fp
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return fp

    identifiers = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            identifiers.append(node.name)
        elif isinstance(node, ast.Name):
            identifiers.append(node.id)

    if identifiers:
        snake = sum(1 for n in identifiers if re.match(r'^[a-z][a-z0-9_]*$', n))
        fp.snake_case_ratio = snake / len(identifiers)
        fp.avg_name_length = sum(len(n) for n in identifiers) / len(identifiers)
        fp.descriptive_names_ratio = sum(1 for n in identifiers if len(n) > 5) / len(identifiers)

    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if functions:
        lengths = [n.end_lineno - n.lineno + 1 for n in functions if n.end_lineno]
        fp.avg_function_length = sum(lengths) / len(lengths) if lengths else 0
        with_docs = sum(1 for f in functions if f.body and isinstance(f.body[0], ast.Expr))
        fp.docstring_coverage = with_docs / len(functions)

    fp.comment_density = (sum(1 for l in lines if l.strip().startswith("#")) / total_lines) * 100
    return fp


def compare_styles(fp1: StyleFingerprint, fp2: StyleFingerprint) -> float:
    """Cosine similarity between two style fingerprints."""
    def vec(fp):
        return [fp.snake_case_ratio, fp.avg_name_length/20, fp.descriptive_names_ratio,
                fp.avg_function_length/50, fp.docstring_coverage, fp.comment_density/20]
    v1, v2 = vec(fp1), vec(fp2)
    dot = sum(a*b for a,b in zip(v1,v2))
    n1 = sum(a**2 for a in v1)**0.5
    n2 = sum(a**2 for a in v2)**0.5
    return dot / (n1 * n2) if n1 and n2 else 0.0
