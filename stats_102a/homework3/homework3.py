import re


# a: Contains at least one digit
q1_a = r".*\d.*"

# b: Exactly 4 letters
q1_b = r"^[A-Za-z]{4}$"

# c: Name pattern (First Last or First Middle Last)
q1_c = r"^[A-Z][a-z]+(\s[A-Z][a-z]*){1,2}$"


# a: 16-digit number starting with 5
q2_a = r"^[5][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}$"

# b: 13–16 digit number starting with 4
q2_b = r"^[4][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\s?[0-9]{1,4}$"


# a: At least one letter + one digit, 8+ chars
q3_a = r"^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]{8,}$"

# b: At least one lowercase, one uppercase, one digit, 8+ chars
q3_b = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])[a-zA-Z0-9]{8,}$"



# a: Hex string (only a–f)
q4_a = r"^[a-f]+$"

# b: Detect repeating 3-character sequence
q4_b = r"^(.{3}).*\1.*$"

# c: Not a full repetition of a smaller substring
q4_c = r"^(?!.*(.+)\1+$).+$"