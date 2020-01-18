"""
Loki - Lock and key-based artificial-life simulation generating pretty patterns.
Copyright (C) 2019 Dylan Banarse

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from flask import Flask, render_template, Response
import loki1Dv as loki

app = Flask(__name__)
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
def gen(loki):
    """Video streaming generator function."""
    while True:
        # frame = loki.step_frame(render_method='rgb_energy_up')
        frame = loki.step_frame(render_method='keys_energy_up')
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    config = loki.get_config(show_resource=True,
            extraction_method='mean', width=128, num_1d_history=240, display='headless')
    # config['gui'] = 'yield_frame'
    return Response(gen(loki.Loki(config)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)

