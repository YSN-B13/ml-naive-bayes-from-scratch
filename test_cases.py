from naive_bayes import naive_bayes

# Example expected Yes
print("Test Case 1: Rain, Mild, Normal, Weak")
print(naive_bayes(outlook="Rain", temp="Mild", humidity="Normal", wind="Weak"))

# Example expected No
print("\nTest Case 2: Sunny, Hot, High, Strong")
print(naive_bayes(outlook="Sunny", temp="Hot", humidity="High", wind="Strong"))
