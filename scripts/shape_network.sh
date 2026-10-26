#!/usr/bin/env bash
# shape_network.sh
# ----------------
# Apply or remove tc-netem + tc-tbf bandwidth shaping on the source
# container's eth0 interface (the pipeline network).
#
# Must be run inside the source container (requires NET_ADMIN capability).
#
# Usage
# -----
#   shape_network.sh set   <bandwidth_kbps> [delay_ms] [jitter_ms]
#   shape_network.sh clear
#   shape_network.sh show
#
# Examples
#   shape_network.sh set 1000          # 1 Mbps, no added delay
#   shape_network.sh set 500 20 5      # 500 Kbps, 20ms Â± 5ms delay
#   shape_network.sh clear             # remove all shaping
#   shape_network.sh show              # display current qdiscs

set -euo pipefail

IFACE="${IFACE:-eth0}"
CMD="${1:-show}"
BW_KBPS="${2:-1000}"
DELAY_MS="${3:-0}"
JITTER_MS="${4:-0}"

# Scale burst dynamically with bandwidth: tbf requires burst >= rate / HZ.
# Assuming 50-100 Hz timer granularity, allocate 20ms of rate, with a 4096-byte floor.
BURST_CALC=$(( (BW_KBPS * 125) / 50 ))
if [[ "$BURST_CALC" -lt 4096 ]]; then
  BURST="${BURST:-4096}"
else
  BURST="${BURST:-$BURST_CALC}"
fi

# Queue limit in bytes — large enough that one full payload (≤1 MB) is never
# dropped while waiting for tokens. Eliminates TCP retransmits and latency spikes.
LIMIT="1048576" # 1 MB

_clear() {
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
}

case "$CMD" in
  set)
    _clear
    BW_BPS=$(( BW_KBPS * 1000 ))

    if [[ "$DELAY_MS" -gt 0 ]]; then
      # netem (delay + jitter) -> tbf (rate limit)
      tc qdisc add dev "$IFACE" root handle 1: netem \
          delay "${DELAY_MS}ms" "${JITTER_MS}ms" distribution normal
      tc qdisc add dev "$IFACE" parent 1:1 handle 10: tbf \
          rate "${BW_KBPS}kbit" burst "$BURST" limit "$LIMIT"
    else
      # Just rate-limit with tbf (rate in kbit/s = kilobits per second)
      tc qdisc add dev "$IFACE" root handle 1: tbf \
          rate "${BW_KBPS}kbit" burst "$BURST" limit "$LIMIT"
    fi

    echo "[shape_network] Applied: ${BW_KBPS} Kbps, delay ${DELAY_MS}ms +- ${JITTER_MS}ms on ${IFACE}"
    ;;

  clear)
    _clear
    echo "[shape_network] Cleared all qdiscs on ${IFACE}"
    ;;

  show)
    tc qdisc show dev "$IFACE"
    tc -s qdisc show dev "$IFACE"
    ;;

  *)
    echo "Usage: $0 set <bw_kbps> [delay_ms] [jitter_ms] | clear | show"
    exit 1
    ;;
esac
