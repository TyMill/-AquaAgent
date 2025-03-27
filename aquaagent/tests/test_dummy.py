
import unittest
from aquaagent.agent import AquaAgent

class TestAquaAgent(unittest.TestCase):
    def test_instantiation(self):
        # Testuje tylko, czy agent się tworzy
        agent = AquaAgent(data_path='test.csv', file_type='csv', mode='interactive')
        self.assertIsNotNone(agent)

if __name__ == '__main__':
    unittest.main()
