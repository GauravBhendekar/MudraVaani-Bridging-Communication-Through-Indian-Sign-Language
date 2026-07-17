from flask import Flask, request
from flask_socketio import SocketIO, join_room, leave_room, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "mudravaani-secret"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


@app.route("/")
def index():
    return "MudraVaani signaling server (HTTPS) running."


#JOIN ROOM
@socketio.on("join")
def handle_join(data):
    room = data.get("room")
    if not room:
        return

    join_room(room)
    sid = request.sid

    emit("joined", {"sid": sid, "room": room})
    emit("peer-joined", {"sid": sid}, to=room, include_self=False)


#SIGNAL EXCHANGE
@socketio.on("signal")
def handle_signal(data):
    room = data.get("room")
    signal = data.get("signal")
    if not room or signal is None:
        return

    emit(
        "signal",
        {"sid": request.sid, "signal": signal},
        to=room,
        include_self=False,
    )


#LEAVE ROOM
@socketio.on("leave")
def handle_leave(data):
    room = data.get("room")
    if room:
        leave_room(room)
        emit("peer-left", {"sid": request.sid}, to=room, include_self=False)


#DISCONNECT
@socketio.on("disconnect")
def handle_disconnect():
    emit("peer-left", {"sid": request.sid}, broadcast=True)


#HTTPS SERVER
if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=True,
        ssl_context=("cert.pem", "key.pem") 
    )
    
#http-server -S -C cert.pem -K key.pem -p 5500
