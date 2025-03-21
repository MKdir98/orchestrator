import argparse
from PIL import Image
from Xlib import display, X
import os
display_num = 1
os.environ["DISPLAY"] = f":{display_num}"
import pyautogui

def capture_screen(output_file):
    """
    Capture a screenshot of the specified display and save it to the output file.
    """

    disp = display.Display(f":{display_num}")
    root = disp.screen().root

    geom = root.get_geometry()
    width, height = geom.width, geom.height

    raw = root.get_image(0, 0, width, height, X.ZPixmap, 0xFFFFFFFF)
    image = Image.frombytes("RGB", (width, height), raw.data, "raw", "BGRX")

    image.save(output_file)
    print(f"Screenshot saved to {output_file}")

def click_mouse(x, y, button='left', clicks=1):
    """
    Simulate a mouse click at the specified coordinates.
    """
    pyautogui.moveTo(x, y)
    if button == 'left':
        pyautogui.click(clicks=clicks)
    elif button == 'right':
        pyautogui.rightClick()
    elif button == 'double':
        pyautogui.doubleClick()
    print(f"Clicked at coordinates ({x}, {y}) with {button} button")

def type_text(text):
    """
    Simulate typing the specified text.
    """
    pyautogui.write(text)
    print(f"Typed: {text}")

def send_key(key):
    """
    Simulate pressing a key or combination of keys.
    """
    pyautogui.press(key)
    print(f"Pressed key: {key}")

def run_command(command):
    """
    Run a shell command and return the result.
    """
    import subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    stdout, stderr = result.stdout, result.stderr
    if stdout and stderr:
        return stdout + "\n" + stderr
    elif stdout or stderr:
        return stdout + stderr
    else:
        return "The command finished running."

def run_background_command(command):
    """
    Run a shell command in the background.
    """
    import subprocess
    subprocess.Popen(command, shell=True)
    return "The command has been started."

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Capture screenshots, click, or type based on actions."
    )
    subparsers = parser.add_subparsers(dest="action", help="Actions to perform")

    # Subparser for screenshot action
    screenshot_parser = subparsers.add_parser("screenshot", help="Take a screenshot")
    screenshot_parser.add_argument(
        "output_file", type=str, help="Output file to save the screenshot"
    )

    # Subparser for click action
    click_parser = subparsers.add_parser("click", help="Simulate a mouse click")
    click_parser.add_argument("x", type=int, help="X coordinate for the click")
    click_parser.add_argument("y", type=int, help="Y coordinate for the click")
    click_parser.add_argument("--button", type=str, default='left', help="Button to click (left, right, double)")
    click_parser.add_argument("--clicks", type=int, default=1, help="Number of clicks")

    # Subparser for type action
    type_parser = subparsers.add_parser("type", help="Simulate typing text")
    type_parser.add_argument("text", type=str, help="Text to type")

    # Subparser for key press action
    key_parser = subparsers.add_parser("key", help="Simulate pressing a key")
    key_parser.add_argument("key", type=str, help="Key or combination to press")

    # Subparser for run command action
    run_parser = subparsers.add_parser("run", help="Run a shell command")
    run_parser.add_argument("command", type=str, help="Shell command to run")

    # Subparser for run background command action
    run_bg_parser = subparsers.add_parser("runbg", help="Run a shell command in the background")
    run_bg_parser.add_argument("command", type=str, help="Shell command to run in the background")

    # Parse arguments
    args = parser.parse_args()

    # Perform actions based on the chosen subcommand
    if args.action == "screenshot":
        capture_screen(args.output_file)
    elif args.action == "click":
        click_mouse(args.x, args.y, args.button, args.clicks)
    elif args.action == "type":
        type_text(args.text)
    elif args.action == "key":
        send_key(args.key)
    elif args.action == "run":
        print(run_command(args.command))
    elif args.action == "runbg":
        print(run_background_command(args.command))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
