import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define pump efficiency
# -------------------------------
def eta_p(Q):
    """Pump efficiency (unitless)"""
    return -0.461*Q**3 - 2.678*Q**2 + 29.433*Q + 2.25

# -------------------------------
# 2. Constants
# -------------------------------
rho = 1000  # kg/m3
g = 9.81    # m/s2
H0 = 50     # fixed head in meters

# -------------------------------
# 3. Flow array
# -------------------------------
Q = np.linspace(0.1, 5, 300)  # focus on small Q range

# -------------------------------
# 4. Compute proxy with H constant
# -------------------------------
# P_elec ∝ Q / eta_p
Pe_proxy = Q / eta_p(Q)  # ignore constants rho*g*H0 since H constant
n_proxy = Q / np.sqrt(Pe_proxy)  # proxy

# -------------------------------
# 5. Find BEP
# -------------------------------
idx_eta = np.argmax(eta_p(Q))
idx_proxy = np.argmax(n_proxy)

Q_BEP_eta = Q[idx_eta]
Q_BEP_proxy = Q[idx_proxy]

eta_max = eta_p(Q)[idx_eta]
proxy_max = n_proxy[idx_proxy]

print(f"BEP from eta_p: Q = {Q_BEP_eta:.2f} m³/h, eta_max = {eta_max:.4f}")
print(f"BEP from proxy: Q = {Q_BEP_proxy:.2f} m³/h, n_proxy_max = {proxy_max:.4f}")

# -------------------------------
# 6. Plot
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(Q, eta_p(Q), label='η_p(Q)', linewidth=2)
plt.plot(Q, n_proxy, '--', label='proxy n_proxy', linewidth=2)
plt.axvline(Q_BEP_eta, color='blue', linestyle=':', label='BEP η_p')
plt.axvline(Q_BEP_proxy, color='orange', linestyle=':', label='BEP proxy')
plt.xlabel('Flow rate Q [m³/h]')
plt.ylabel('Efficiency / Proxy (unitless)')
plt.title('Pump efficiency vs proxy (H constant)')
plt.legend()
plt.grid(True)
plt.show()
