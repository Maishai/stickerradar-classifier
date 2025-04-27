# Sticker Classifier

This binary classifier is designed to identify whether a given image contains a sticker or not.

## Example Call

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "x-api-key: changeme" \
  -d '{"image_base64": "'"$(base64 -w 0 uploads/test_images/stickers/img1.jpg)"'"}'
```

```json
{ "sticker_probability": 0.9998990297317505 }
```
