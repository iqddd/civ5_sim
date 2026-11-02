"""Browser check: version colors and explicit synchronization of an open replay."""

import argparse
from copy import deepcopy
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(url, screenshot):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1050})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        catalog = page.request.get(url.rstrip('/')+'/api/matches').json()[:2]
        payload = page.request.get(url.rstrip('/')+'/api/replay', params={'id': catalog[0]['id']}).json()
        catalog[0].update(refresh_status='pending', port='AI/0.9.2', revision='old')
        catalog[1].update(refresh_status='updated', port='AI/0.9.3', revision='other')
        old = deepcopy(payload)
        old['provenance'] = {'port': 'AI/0.9.2', 'revision': 'old', 'refresh_status': 'pending'}
        updated = deepcopy(payload)
        updated['provenance'] = {'port': 'AI/0.9.3', 'revision': 'new', 'refresh_status': 'updated'}
        updated['name'] = 'reloaded-payload'
        updated['steps'] = updated['steps'][:2]
        updated['frames'] = updated['frames'][:3]
        updated['turns'] = [t for t in updated['turns'] if t['frame'] <= 2]
        state = {'published': False}
        page.route('**/api/matches', lambda route: route.fulfill(json=catalog))
        page.route('**/api/status', lambda route: route.fulfill(json=None))
        page.route('**/api/replay?*', lambda route: route.fulfill(
            json=updated if state['published'] else old))
        from urllib.parse import quote
        page.goto(url.rstrip('/')+'/#'+quote(catalog[0]['id'], safe=''))
        page.locator('#verified').wait_for(state='visible')
        assert page.locator('.match.pending').count() == 1
        assert page.locator('.match.updated').count() == 1
        page.evaluate('seek(5)')
        assert page.evaluate('index') == 5
        state['published'] = True
        catalog[0].update(refresh_status='updated', port='AI/0.9.3', revision='new')
        page.evaluate('refreshCatalog()')
        page.wait_for_function("document.getElementById('openedVersion').textContent.includes('есть новый реплей')")
        assert page.evaluate('data.provenance.revision') == 'old'
        page.click('#refresh')
        page.wait_for_function("data.provenance.revision==='new'")
        assert page.evaluate('data.name') == 'reloaded-payload'
        assert page.evaluate('index') == 2  # The replacement can be shorter.
        assert '0.9.3' in page.locator('#openedVersion').inner_text()
        assert 'есть новый' not in page.locator('#openedVersion').inner_text()
        assert page.locator('.match.updated').count() == 2
        assert not errors, errors
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(screenshot), full_page=True)
        browser.close()
        print('Refresh: pending/updated colors, stale-open notice, reload and frame clamp passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('url')
    parser.add_argument('--screenshot', type=Path, default=Path('build/combat-refresh-check.png'))
    args = parser.parse_args()
    check(args.url, args.screenshot)
