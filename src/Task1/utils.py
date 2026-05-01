# utils.py

def format_inr(value: float) -> str:
    return f"₹{value:,.2f}"

def parse_percentage(value: str) -> float:
    return float(value.strip().replace("%", ""))

def get_valid_number(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid number. Try again.")

def get_valid_percentage(prompt: str) -> float:
    while True:
        try:
            val = parse_percentage(input(prompt))
            if val < 0:
                print("Cannot be negative.")
                continue
            return val
        except Exception:
            print("Invalid input. Example: 40 or 40%")

def get_valid_crash(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()

        if raw == "":
            return 0.0

        try:
            val = parse_percentage(raw)

            if val > 0:
                print("Warning: crash should be negative. Converting.")
                val = -abs(val)

            if val < -100:
                print("Crash cannot exceed -100%. Capping at -100%.")
                val = -100

            return val

        except Exception:
            print("Invalid crash value. Example: -40 or -25%")