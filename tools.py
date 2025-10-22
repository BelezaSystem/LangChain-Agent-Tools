from langchain_core.tools import StructuredTool


class MathTools:   

    def __init__(self, factor: float = 1.0):
        self.factor = factor

    def multiply(self, a: float, b: float) -> float:
        """Multiplica a * b."""
        return (a * b) * self.factor

    def subtract(self, a: float, b: float) -> float:
        """Subtrai a - b."""
        return (a - b) * self.factor

    def as_tools(self):
        """Retorna a lista de tools derivadas dos métodos da classe."""
        return [
            StructuredTool.from_function(
                self.multiply,
                name="multiply",
            ),
            StructuredTool.from_function(
                self.subtract,
                name="subtract",
            ),
        ]


class IdentityTools:
    """Ferramentas para coletar dados do usuário e calcular idade."""

    def collect_dob(self) -> str:
        """Solicita data de nascimento (dd/mm/yyyy)."""
        return (
            "Por favor, informe sua data de nascimento no formato dd/mm/yyyy."
        )

    def compute_age(self, dob: str) -> str:
        """Calcula idade por dd/mm/yyyy."""
        from datetime import date, datetime
        try:
            birth = datetime.strptime(dob, "%d/%m/%Y").date()
        except Exception:
            return (
                "Data inválida. Use o formato dd/mm/yyyy (ex.: 17/05/1990)."
            )
        today = date.today()
        age = today.year - birth.year - (
            (today.month, today.day) < (birth.month, birth.day)
        )
        return f"Sua idade é {age} anos."

    def as_tools(self):
        """Retorna a lista de tools para coleta e cálculo de idade."""
        return [
            StructuredTool.from_function(
                self.collect_dob,
                name="collect_dob",
            ),
            StructuredTool.from_function(
                self.compute_age,
                name="compute_age",
            ),
        ]