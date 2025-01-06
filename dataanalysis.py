import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"  # Replace with the correct path
data = pd.read_csv(file_path)

# Filter for attack data (Label != 'BENIGN')
attack_data = data[data[' Label'] != 'BENIGN']

# Compute key indicators
attack_indicators = {
    "SYN Flood": {
        "high_syn_flag": attack_data["SYN Flag Count"].mean(),
        "low_ack_flag": attack_data["ACK Flag Count"].mean(),
        "high_fwd_packets": attack_data["Total Fwd Packets"].mean(),
        "low_bwd_packets": attack_data["Total Backward Packets"].mean(),
    },
    "HTTP Flood": {
        "long_duration": attack_data["Flow Duration"].mean(),
        "high_fwd_packets": attack_data["Total Fwd Packets"].mean(),
        "high_fwd_length": attack_data["Total Length of Fwd Packets"].mean(),
    },
    "UDP Flood": {
        "high_packets_per_sec": attack_data["Flow Packets/s"].mean(),
        "low_bwd_packets": attack_data["Total Backward Packets"].mean(),
        "uniform_packet_size": attack_data["Packet Length Std"].mean(),
    },
    "ICMP Flood": {
        "high_packets_per_sec": attack_data["Flow Packets/s"].mean(),
        "low_iat_mean": attack_data["Flow IAT Mean"].mean(),
        "uniform_packet_size": attack_data["Packet Length Std"].mean(),
    },
}

# Decision Rules
def decide_attack_type(attack_indicators):
    decision = {}
    
    # SYN Flood
    if (
        attack_indicators["SYN Flood"]["high_syn_flag"] > 10 and
        attack_indicators["SYN Flood"]["low_ack_flag"] < 5 and
        attack_indicators["SYN Flood"]["high_fwd_packets"] > 100 and
        attack_indicators["SYN Flood"]["low_bwd_packets"] < 10
    ):
        decision["SYN Flood"] = True
    else:
        decision["SYN Flood"] = False

    # HTTP Flood
    if (
        attack_indicators["HTTP Flood"]["long_duration"] > 50000 and
        attack_indicators["HTTP Flood"]["high_fwd_packets"] > 100 and
        attack_indicators["HTTP Flood"]["high_fwd_length"] > 1000
    ):
        decision["HTTP Flood"] = True
    else:
        decision["HTTP Flood"] = False

    # UDP Flood
    if (
        attack_indicators["UDP Flood"]["high_packets_per_sec"] > 1000 and
        attack_indicators["UDP Flood"]["low_bwd_packets"] < 10 and
        attack_indicators["UDP Flood"]["uniform_packet_size"] < 50
    ):
        decision["UDP Flood"] = True
    else:
        decision["UDP Flood"] = False

    # ICMP Flood
    if (
        attack_indicators["ICMP Flood"]["high_packets_per_sec"] > 1000 and
        attack_indicators["ICMP Flood"]["low_iat_mean"] < 1000 and
        attack_indicators["ICMP Flood"]["uniform_packet_size"] < 50
    ):
        decision["ICMP Flood"] = True
    else:
        decision["ICMP Flood"] = False

    return decision

# Determine the attack type
decision = decide_attack_type(attack_indicators)

# Display the decision
print("Attack Type Decisions:")
for attack, detected in decision.items():
    print(f"{attack}: {'Detected' if detected else 'Not Detected'}")

# Visualize indicators for each attack type
for attack_type, indicators in attack_indicators.items():
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(indicators.keys()), y=list(indicators.values()))
    plt.title(f"Indicators for {attack_type}")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=45)
    plt.show()
