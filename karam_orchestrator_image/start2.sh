#!/usr/bin/env bash
set -euo pipefail
set -x

export DISPLAY=${DISPLAY:-:99}
export NOVNC_DIR=${NOVNC_DIR:-/opt/novnc}

# پاک کردن lock files قدیمی اگر وجود دارند
rm -f /tmp/.X${DISPLAY#:}-lock
rm -f /tmp/.X11-unix/X${DISPLAY#:}

# اجرای Xvfb
Xvfb $DISPLAY -screen 0 1920x1080x24 -nolisten tcp &
XVFB_PID=$!
sleep 2

# اجرای D-Bus
eval "$(dbus-launch --sh-syntax)"
export DBUS_SESSION_BUS_ADDRESS

# ریست پنل Mate (اختیاری)
dconf reset -f /org/mate/panel/

# اجرای دسکتاپ کامل Mate داخل background
nohup mate-session >/var/log/mate-session.log 2>&1 &
sleep 5  # صبر برای آماده شدن Mate

# فعال کردن نمایش آیکون‌ها
gsettings set org.mate.background show-desktop-icons true

# اجرای فایل منیجر داخل Mate session (بعد از آماده شدن)
nohup caja --force-desktop >/dev/null 2>&1 &

# اجرای VNC روی پورت 5901 با حالت shared (چند client همزمان)
nohup x11vnc -display $DISPLAY -forever -nopw -listen 0.0.0.0 -rfbport 5901 -shared >/var/log/x11vnc.log 2>&1 &

# اجرای noVNC روی پورت 6080
nohup $NOVNC_DIR/utils/novnc_proxy --vnc 0.0.0.0:5901 --listen 0.0.0.0:6080 >/var/log/novnc.log 2>&1 &

# اجرای nginx در foreground
nginx -g "daemon off;"
