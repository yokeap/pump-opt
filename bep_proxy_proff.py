import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define pump models
# -------------------------------
def H(Q):
    """Pump head in meters, Q in m3/h"""
    return -1.404*Q**2 - 0.909*Q + 50.914

def eta_p(Q):
    """Pump efficiency in fraction (0-1), Q in m3/h"""
    return (-0.461*Q**3 - 2.678*Q**2 + 29.433*Q + 2.25)/100

# -------------------------------
# 2. Constants
# -------------------------------
rho = 1000  # kg/m3
g = 9.81    # m/s2

# -------------------------------
# 3. Flow array
# -------------------------------
Q = np.linspace(0.1, 30, 300)  # avoid 0 to prevent div by zero

# -------------------------------
# 4. Compute hydraulic and electric power
# -------------------------------
Q_m3s = Q / 3600  # convert m3/h to m3/s
P_hyd = rho * g * Q_m3s * H(Q)        # hydraulic power in Watts
Pe = P_hyd / eta_p(Q)                  # electrical power in Watts

# -------------------------------
# 5. Compute proxy
# -------------------------------
n_proxy = Q / np.sqrt(Pe)  # proxy

# Normalize for plotting
eta_norm = eta_p(Q)/np.max(eta_p(Q))
n_proxy_norm = n_proxy/np.max(n_proxy)

# -------------------------------
# 6. Find BEP
# -------------------------------
idx_eta = np.argmax(eta_p(Q))
idx_proxy = np.argmax(n_proxy)

Q_BEP_eta = Q[idx_eta]
Q_BEP_proxy = Q[idx_proxy]

eta_max = eta_p(Q)[idx_eta]
proxy_max = n_proxy[idx_proxy]

print(f"BEP from eta_p: Q = {Q_BEP_eta:.2f} m³/h, eta_max = {eta_max*100:.2f} %")
print(f"BEP from proxy: Q = {Q_BEP_proxy:.2f} m³/h, n_proxy_max = {proxy_max:.4f}")

# -------------------------------
# 7. Plot
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(Q, eta_norm, label='Normalized $\eta_p(Q)$', linewidth=2)
plt.plot(Q, n_proxy_norm, '--', label='Normalized proxy $n_{proxy}$', linewidth=2)
plt.axvline(Q_BEP_eta, color='blue', linestyle=':', label='BEP η_p')
plt.axvline(Q_BEP_proxy, color='orange', linestyle=':', label='BEP proxy')
plt.xlabel('Flow rate Q [m³/h]')
plt.ylabel('Normalized efficiency / proxy')
plt.title('Comparison of pump efficiency vs proxy with BEP')
plt.legend()
plt.grid(True)
plt.show()
