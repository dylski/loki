import buttonshim

BUTTONS = [buttonshim.BUTTON_A, buttonshim.BUTTON_B, buttonshim.BUTTON_C, buttonshim.BUTTON_D, buttonshim.BUTTON_E]

@buttonshim.on_release(BUTTONS)
def button_r_handler(button, pressed):
  global keycode
  keycode = button

def last_button_release():
  global keycode
  button = keycode
  keycode = -1
  return button

keycode  = -1
