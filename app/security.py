import re


class InputSanitizer:
    """Apply lightweight input checks; these are not a security boundary."""

    INJECTION_PATTERNS = (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"new\s+instructions:",
        r"system\s*prompt",
        r"---\s*end\s*(of)?\s*prompt",
        r"pretend\s+you\s+are",
        r"act\s+as\s+(if\s+)?you",
        r"bypass\s+(all\s+)?restrictions",
    )

    def __init__(self) -> None:
        self.patterns = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INJECTION_PATTERNS
        )

    def check(self, text: str) -> tuple[bool, str | None]:
        if any(pattern.search(text) for pattern in self.patterns):
            return False, "Potential prompt injection detected."
        return True, None

    @staticmethod
    def clean(text: str) -> str:
        text = re.sub(r"[-]{3,}|[=]{3,}|[*]{3,}", "", text)
        return text.replace("{{", "{ {").replace("}}", "} }").strip()


class PIIDetector:
    PATTERNS = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b(?:\d[ -]*?){13,19}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    }

    MASKS = {
        "ssn": "[REDACTED SSN]",
        "credit_card": "[REDACTED CREDIT CARD]",
        "phone": "[REDACTED PHONE]",
        "email": "[REDACTED EMAIL]",
    }

    def __init__(self) -> None:
        self.patterns = {
            kind: re.compile(pattern)
            for kind, pattern in self.PATTERNS.items()
        }

    def detect(self, text: str) -> dict[str, list[str]]:
        return {
            kind: matches
            for kind, pattern in self.patterns.items()
            if (matches := pattern.findall(text))
        }

    def mask(self, text: str) -> str:
        for kind, pattern in self.patterns.items():
            text = pattern.sub(self.MASKS[kind], text)
        return text


class OutputValidator:
    def __init__(self, pii_detector: PIIDetector) -> None:
        self.pii_detector = pii_detector

    def validate(self, output: str) -> tuple[str, list[str]]:
        pii_found = self.pii_detector.detect(output)
        if not pii_found:
            return output, []

        masked_output = self.pii_detector.mask(output)
        return masked_output, [
            f"PII masked in output: {', '.join(sorted(pii_found))}"
        ]


class SecurityPipeline:
    def __init__(self) -> None:
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator(self.pii_detector)

    def check_input(self, user_input: str) -> tuple[bool, str, list[str]]:
        is_safe, warning = self.sanitizer.check(user_input)
        if not is_safe:
            return False, "", [warning] if warning else []

        sanitized_input = self.sanitizer.clean(user_input)
        pii_found = self.pii_detector.detect(sanitized_input)
        if pii_found:
            sanitized_input = self.pii_detector.mask(sanitized_input)
            notes = [
                f"PII detected and masked: {', '.join(sorted(pii_found))}"
            ]
        else:
            notes = []
        return True, sanitized_input, notes

    def check_output(self, model_output: str) -> tuple[str, list[str]]:
        return self.output_validator.validate(model_output)
