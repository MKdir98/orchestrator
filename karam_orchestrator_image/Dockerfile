FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV GTK_MODULES=gail:atk-bridge
ENV QT_ACCESSIBILITY=1

RUN apt-get update && apt-get install -y \
  # X/desktop & VNC
  xvfb x11vnc openbox lxpanel pcmanfm dbus-x11 \
  # AT-SPI and Python GI bindings
  at-spi2-core python3 python3-gi gir1.2-atspi-2.0 \
  # Tools expected by gui apps
  wmctrl xdotool imagemagick scrot tesseract-ocr \
  # Python libs
  python3-pip python3-numpy python3-opencv python3-pil \
  # misc
  fonts-dejavu-core wget ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

# Python OCR
RUN pip3 install --no-cache-dir pytesseract

# Install noVNC + websockify
RUN git clone https://github.com/novnc/noVNC.git /opt/novnc && \
    git clone https://github.com/novnc/websockify /opt/novnc/utils/websockify && \
    ln -s /opt/novnc/vnc.html /opt/novnc/index.html

COPY . /app
WORKDIR /app

# Entrypoint
COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

EXPOSE 5901 6080
ENTRYPOINT ["/usr/local/bin/start.sh"]
