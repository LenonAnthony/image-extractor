import base64
from pathlib import Path
from tempfile import NamedTemporaryFile
from image_extractor.service.text_extraction import convert_base64


# Test when a valid image file is given
def test_convert_base64_with_valid_image():

    bytes_fake = b"fake_image_data"
    # Create a temporary file with some image-like content
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(bytes_fake)
        temp_path = Path(temp_file.name)

    # Calculate the expected base64 result
    expected_result = base64.b64encode(bytes_fake).decode("utf-8")

    result = convert_base64(temp_path)
    assert result == expected_result

    # Clean up the temporary file
    temp_path.unlink()
