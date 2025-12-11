import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================
# Parámetros generales
# =========================
G = 1.0     # Constante de gravitación (puedes cambiarla)
dt = 0.01   # Paso de tiempo
t_max = 20  # Tiempo total de simulación

# =========================
# Masas y condiciones iniciales
# =========================
m1, m2, m3 = 10.0, 10.0, 10.0

# Posiciones
r1 = np.array([-5.0, 0.0])
r2 = np.array([5.0, 0.0])
r3 = np.array([0.0, 5.0])

# Velocidades iniciales
v1 = np.array([0.0, 0.5])
v2 = np.array([0.0, -0.3])
v3 = np.array([1.6, 0.0])

# =========================
# Funciones auxiliares
# =========================
def aceleraciones(r1, r2, r3, m1, m2, m3):
    def force(p1, p2, m2):
        r = p1 - p2
        dist = np.linalg.norm(r)
        if dist == 0:  # proteger contra división por cero
            return np.zeros(2)
        return -G * m2 * r / dist**3

    a1 = force(r1, r2, m2) + force(r1, r3, m3)
    a2 = force(r2, r1, m1) + force(r2, r3, m3)
    a3 = force(r3, r1, m1) + force(r3, r2, m2)
    return a1, a2, a3

# =========================
# Integración (RK4)
# =========================
def rk4_step(r1, r2, r3, v1, v2, v3, dt):
    # Derivadas iniciales
    a1, a2, a3 = aceleraciones(r1, r2, r3, m1, m2, m3)

    # k1
    k1_v1 = a1 * dt
    k1_v2 = a2 * dt
    k1_v3 = a3 * dt
    k1_r1 = v1 * dt
    k1_r2 = v2 * dt
    k1_r3 = v3 * dt

    # punto medio 1
    r1_mid = r1 + k1_r1 / 2
    r2_mid = r2 + k1_r2 / 2
    r3_mid = r3 + k1_r3 / 2
    v1_mid = v1 + k1_v1 / 2
    v2_mid = v2 + k1_v2 / 2
    v3_mid = v3 + k1_v3 / 2

    a1_mid, a2_mid, a3_mid = aceleraciones(r1_mid, r2_mid, r3_mid, m1, m2, m3)

    # k2
    k2_v1 = a1_mid * dt
    k2_v2 = a2_mid * dt
    k2_v3 = a3_mid * dt
    k2_r1 = v1_mid * dt
    k2_r2 = v2_mid * dt
    k2_r3 = v3_mid * dt

    # punto medio 2
    r1_mid2 = r1 + k2_r1 / 2
    r2_mid2 = r2 + k2_r2 / 2
    r3_mid2 = r3 + k2_r3 / 2
    v1_mid2 = v1 + k2_v1 / 2
    v2_mid2 = v2 + k2_v2 / 2
    v3_mid2 = v3 + k2_v3 / 2

    a1_mid2, a2_mid2, a3_mid2 = aceleraciones(r1_mid2, r2_mid2, r3_mid2, m1, m2, m3)

    # k3
    k3_v1 = a1_mid2 * dt
    k3_v2 = a2_mid2 * dt
    k3_v3 = a3_mid2 * dt
    k3_r1 = v1_mid2 * dt
    k3_r2 = v2_mid2 * dt
    k3_r3 = v3_mid2 * dt

    # punto final
    r1_end = r1 + k3_r1
    r2_end = r2 + k3_r2
    r3_end = r3 + k3_r3
    v1_end = v1 + k3_v1
    v2_end = v2 + k3_v2
    v3_end = v3 + k3_v3

    a1_end, a2_end, a3_end = aceleraciones(r1_end, r2_end, r3_end, m1, m2, m3)

    # k4
    k4_v1 = a1_end * dt
    k4_v2 = a2_end * dt
    k4_v3 = a3_end * dt
    k4_r1 = v1_end * dt
    k4_r2 = v2_end * dt
    k4_r3 = v3_end * dt

    # Combinar
    r1_new = r1 + (k1_r1 + 2*k2_r1 + 2*k3_r1 + k4_r1)/6
    r2_new = r2 + (k1_r2 + 2*k2_r2 + 2*k3_r2 + k4_r2)/6
    r3_new = r3 + (k1_r3 + 2*k2_r3 + 2*k3_r3 + k4_r3)/6

    v1_new = v1 + (k1_v1 + 2*k2_v1 + 2*k3_v1 + k4_v1)/6
    v2_new = v2 + (k1_v2 + 2*k2_v2 + 2*k3_v2 + k4_v2)/6
    v3_new = v3 + (k1_v3 + 2*k2_v3 + 2*k3_v3 + k4_v3)/6

    return r1_new, r2_new, r3_new, v1_new, v2_new, v3_new

# =========================
# Prepara datos para animar
# =========================
steps = int(t_max / dt)
traj1, traj2, traj3 = [], [], []

# Iniciales
pos1, pos2, pos3 = r1.copy(), r2.copy(), r3.copy()
vel1, vel2, vel3 = v1.copy(), v2.copy(), v3.copy()

for _ in range(steps):
    traj1.append(pos1.copy())
    traj2.append(pos2.copy())
    traj3.append(pos3.copy())
    pos1, pos2, pos3, vel1, vel2, vel3 = rk4_step(pos1, pos2, pos3, vel1, vel2, vel3, dt)

traj1 = np.array(traj1)
traj2 = np.array(traj2)
traj3 = np.array(traj3)

# =========================
# Configuración de la animación
# =========================
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_aspect('equal')

line1, = ax.plot([], [], 'r-', label="Cuerpo 1")
line2, = ax.plot([], [], 'g-', label="Cuerpo 2")
line3, = ax.plot([], [], 'b-', label="Cuerpo 3")
dot1,  = ax.plot([], [], 'ro')
dot2,  = ax.plot([], [], 'go')
dot3,  = ax.plot([], [], 'bo')

ax.legend()

def update(frame):
    line1.set_data(traj1[:frame,0], traj1[:frame,1])
    line2.set_data(traj2[:frame,0], traj2[:frame,1])
    line3.set_data(traj3[:frame,0], traj3[:frame,1])
    dot1.set_data([traj1[frame,0]], [traj1[frame,1]])
    dot2.set_data([traj2[frame,0]], [traj2[frame,1]])
    dot3.set_data([traj3[frame,0]], [traj3[frame,1]])
    return line1, line2, line3, dot1, dot2, dot3

def init():
    dot1.set_data([], [])
    dot2.set_data([], [])
    dot3.set_data([], [])
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3, dot1, dot2, dot3

ani = FuncAnimation(fig, update, frames=steps, init_func=init,
                    interval=30, blit=True)

from matplotlib.animation import PillowWriter

ani.save("three_body.gif", writer=PillowWriter(fps=30))
