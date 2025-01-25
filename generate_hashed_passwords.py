import streamlit_authenticator as stauth

def main():
    print("Enter the plain-text passwords you want to hash.")
    print("Type 'exit' to stop adding passwords.\n")

    passwords = []
    while True:
        password = input("Enter a password: ")
        if password.lower() == 'exit':
            break
        passwords.append(password)

    if passwords:
        print("\nHashing passwords...")
        # Use the updated hash method
        hashed_passwords = [stauth.Hasher.hash(pw) for pw in passwords]
        print("\nHere are the hashed passwords:")
        for i, hashed_password in enumerate(hashed_passwords):
            print(f"Password {i+1}: {hashed_password}")

if __name__ == "__main__":
    main()
