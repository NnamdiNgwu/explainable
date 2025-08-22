# explainable_architectural_transfer
```bash
# Example curl command to test the /api/predict endpoint with an event payload

curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d 'event = {
    "src_ip": "192.168.1.10",
    "dst_ip": "10.0.0.5",
    "src_port": 443,
    "dst_port": 8080,
    "protocol": "TCP",
    "timestamp": "2025-08-22T14:32:00Z",
    "megabytes_sent": 1.2,
    "megabytes_received": 0.3,
    "post_burst": True,
    "destination_entropy": 0.85,
    "hour": 14,
    "first_time_destination": False,
    "after_hours": False,
    # ... add all other features expected by your models ...
}
```
- Replace `http://localhost:5000/api/predict` with the actual endpoint if different.
- Adjust the event fields (`feature1`, `feature2`, `feature3`, etc.) to match your model’s expected input.
- You can add or remove events in the array as needed to test batch processing.

If you have an explanation endpoint, you might do something similar:

```bash
curl -X POST http://localhost:5000/api/explain \
  -H "Content-Type: application/json" \
 
      }'
```
