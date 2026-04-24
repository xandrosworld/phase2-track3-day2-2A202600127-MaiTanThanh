import shutil
import tempfile
import unittest
from pathlib import Path

from memory_agent import MultiMemoryAgent


class MemoryAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_allergy_conflict_uses_newest_fact(self) -> None:
        agent = MultiMemoryAgent(data_dir=self.tmpdir)

        agent.receive("Tôi dị ứng sữa bò.")
        agent.receive("À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.")
        response, state = agent.receive("Tôi dị ứng gì?")

        self.assertEqual(state["user_profile"]["allergy"], "đậu nành")
        self.assertIn("đậu nành", response)

    def test_question_does_not_overwrite_profile_fact(self) -> None:
        agent = MultiMemoryAgent(data_dir=self.tmpdir)

        agent.receive("Tôi dị ứng đậu nành.")
        agent.receive("Tôi dị ứng gì?")

        self.assertEqual(agent.profile.retrieve()["allergy"], "đậu nành")


if __name__ == "__main__":
    unittest.main()

