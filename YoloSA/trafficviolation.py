import subprocess

def helmet_detection():
    subprocess.run(['python', 'helmetdetectionyolo.py'])

def triple_detection():
    subprocess.run(['python', 'triple.py'])

def red_light_running():
    subprocess.run(['python', 'redlightyolo2.py'])

def phone_while_driving():
    subprocess.run(['python', 'yolo.py'])

def main_menu():
    while True:
        print("\nChoose a program to execute:")
        print("1. Helmet Detection")
        print("2. Triple Detection")
        print("3. Red Light Running Detection")
        print("4. Phone While Driving Detection")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            helmet_detection()
        elif choice == '2':
            triple_detection()
        elif choice == '3':
            red_light_running()
        elif choice == '4':
            phone_while_driving()
        elif choice == '5':
            print("Exiting program.")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main_menu()
