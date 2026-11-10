"""Student Manager: облік студентів, профілювання й оптимізація (ЛР9)."""

from student_manager.models import Student
from student_manager.services import StudentService

__version__ = "0.9.0"

__all__ = ["Student", "StudentService", "__version__"]
