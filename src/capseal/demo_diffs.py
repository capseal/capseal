"""Synthetic demo diffs used by quickstart."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoDiff:
    description: str
    content: str
    expected_risk: str = "high"


DEMO_DIFFS: list[DemoDiff] = [
    DemoDiff(
        description="adds subprocess.run() with shell=True to auth module",
        content=(
            "--- a/auth/handler.py\n"
            "+++ b/auth/handler.py\n"
            "@@ -1,4 +1,9 @@\n"
            "+import os\n"
            "+import subprocess\n"
            "+\n"
            " class AuthHandler:\n"
            "     def validate(self, token):\n"
            "-        return self.check_token(token)\n"
            "+        cmd = os.environ.get(\"AUTH_CMD\", \"validate\")\n"
            "+        return subprocess.run(cmd, shell=True, capture_output=True)\n"
        ),
    ),
    DemoDiff(
        description="removes input validation from API endpoint",
        content=(
            "--- a/api/views.py\n"
            "+++ b/api/views.py\n"
            "@@ -10,8 +10,6 @@\n"
            " class UserView:\n"
            "     def update(self, request):\n"
            "-        if not self.validate_input(request.data):\n"
            "-            raise ValidationError(\"Invalid input\")\n"
            "         user = User.objects.get(id=request.data['id'])\n"
            "         user.name = request.data['name']\n"
            "+        user.is_admin = request.data.get('is_admin', False)\n"
            "         user.save()\n"
        ),
    ),
]


__all__ = ["DemoDiff", "DEMO_DIFFS"]
