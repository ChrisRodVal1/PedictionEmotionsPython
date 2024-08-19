import cv2
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, render_template, Response

modelo_final = load_model('C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionEmociones/terceiro_modelo.h5')

app = Flask(__name__, template_folder='templates')

dict_emociones = {
    0: "Enojado",
    1: "Disgustado",
    2: "Miedo",
    3: "Feliz",
    4: "Natural",
    5: "Triste",
    6: "Sorprendido"
}

bounding_box = cv2.CascadeClassifier('C:/Users/HP/Desktop/InteligenciaArtificialCRV/PrediccionEmociones/haarcascade_frontalface_default.xml')

def frames1():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame1 = cap.read()
        frame2 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(frame2, scaleFactor=1.1, minNeighbors=2)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (0, 0, 255), 2)
            roi_frame = frame2[y:y + h, x:x + w]
            img_redi = np.expand_dims(np.expand_dims(cv2.resize(roi_frame, (48, 48)), -1), 0)
            emotion_prediction = modelo_final.predict(img_redi)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame1, dict_emociones[maxindex], (x+20, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame1)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/detecta', methods=['GET'])
def detecta():
    return Response(frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
