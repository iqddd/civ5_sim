"""Browser checks for the local match viewer (installed Edge + Playwright)."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(url, screenshot):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1050}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.locator("#verified").wait_for(state="visible")
        assert page.locator(".match").count() > 0
        assert page.locator(".unit").count() > 0
        page.locator(".unit").first.click()
        assert "Здоровье" in page.locator("#details").inner_text()
        page.click("#next")
        assert page.locator("#timeline").input_value() == "1"
        page.click("#prev")
        assert page.locator("#timeline").input_value() == "0"
        page.click("#nextTurn")
        assert page.locator("#timeline").input_value() != "0"
        page.click("#first")
        page.click("#play")
        page.wait_for_function("Number(document.getElementById('timeline').value)>0")
        page.click("#play")
        paused = page.locator("#timeline").input_value()
        page.wait_for_timeout(750)
        assert page.locator("#timeline").input_value() == paused
        page.click("#last")
        label = page.locator("#turnLabel").inner_text()
        if page.evaluate("Boolean(data.experiment)"):
            assert "Конец найденного плана" in label
        else:
            assert "Матч завершён" in label
        page.click("#first")
        page.keyboard.press("ArrowRight")
        assert page.locator("#timeline").input_value() == "1"
        page.keyboard.press("ArrowLeft")
        assert page.locator("#timeline").input_value() == "0"
        viewbox = page.locator("#map").get_attribute("viewBox")
        page.click("#zoomIn")
        assert page.locator("#map").get_attribute("viewBox") != viewbox
        page.click("#zoomReset")
        assert page.locator("#map").get_attribute("viewBox") == viewbox
        page.check("#coords")
        assert page.locator("#map text").count() >= page.evaluate("data.map.width*data.map.height")
        catalog = page.request.get(url.rstrip("/") + "/api/matches").json()
        fog_match = next((m for m in catalog if m["fog"]), None)
        if fog_match:
            page.get_by_title(fog_match["name"], exact=True).click()
            page.wait_for_function("id=>currentId===id", arg=fog_match["id"])
            page.select_option("#vision", "0")
            expected = page.evaluate("data.roster.filter(u=>{const row=data.frames[index].units[u.id];return row[1]>0&&(u.side===0||data.frames[index].visibility[0][row[0]]==='1')}).length")
            assert page.locator(".unit").count() == expected
            page.select_option("#vision", "all")
            assert page.locator(".unit").count() >= expected
        page.click("#next")
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth"), "Mobile overflow"
        page.screenshot(path=str(screenshot.with_stem(screenshot.stem + "-mobile")), full_page=True)
        page.fill("#search", "there-is-no-such-match")
        assert page.locator(".match").count() == 0
        page.fill("#search", "")
        assert page.locator(".match").count() > 0
        assert not errors, errors
        print("Viewer: stepping, turns, playback/pause, keyboard, zoom, selection, fog, search, mobile; no JS errors")
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--screenshot", type=Path, default=Path("build/combat-viewer-preview.png"))
    args = parser.parse_args()
    check(args.url, args.screenshot)
