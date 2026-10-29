"""Browser smoke check using installed Edge; optional playwright dependency."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(path):
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(
            viewport={"width": 1440, "height": 1000}, device_scale_factor=1
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.resolve().as_uri())
        assert page.locator("#select option").count() == 11
        assert page.locator("polygon").count() == 80 * 52
        page.select_option("#select", "1")
        assert page.locator("polygon").count() == 256
        page.check("#coords")
        page.check("#spawns")
        assert page.locator("circle").count() >= 20
        page.locator("polygon").nth(130).click()
        assert "Клетка" in page.locator("#details").inner_text()
        with page.expect_download() as download:
            page.click("#download")
        assert download.value.suggested_filename == "map-1.json"
        page.screenshot(path=str(path.parent / "preview.png"), full_page=True)
        assert not errors, errors
        print(
            "Viewer: 11 maps, 256 crop hexes, coordinates, spawns, tile details, JSON download; no JS errors"
        )
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("viewer", type=Path)
    check(parser.parse_args().viewer)
