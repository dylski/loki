from flask import Flask, render_template, Response
# Raspberry Pi camera module (requires picamera package, developed by Miguel Grinberg)
import loki1Dv as loki

app = Flask(__name__)
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
def gen(loki):
    """Video streaming generator function."""
    while True:
        frame = loki.step_frame(render_method='keys_energy_up')
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    config = loki.get_config(show_resource=True, render_method='res_energy_up',
            width=64, num_1d_history=48, display='headless')
    # config['gui'] = 'yield_frame'
    return Response(gen(loki.Loki(config)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)

