import unittest
import cryptogram_final
# The function or class you want to test
def check_weights(w1=[],w2=[],w3=[]):
    if len(w1)==len(w2) and len(w2)==len(w3):
        return 1
    else:
        return 0;

# Define a test class that inherits from unittest.TestCase
class TestAddNumbers(unittest.TestCase):
    def test_add_positive_numbers(self):
        result = check_weights(cryptogram_final.w1,cryptogram_final.w2,cryptogram_final.w3)
        self.assertEqual(result, 1)  # Assert that the result is equal to 7


# Run the tests
if __name__ == '__main__':
    unittest.main()
