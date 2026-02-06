from server.flask_app import create_app


def test_health_and_ready_endpoints():
    app = create_app({"TESTING": True})
    client = app.test_client()

    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'ok'

    ready = client.get('/ready')
    assert ready.status_code == 200
    body = ready.get_json()
    assert body['status'] in {'ready', 'not_ready'}
    assert 'checks' in body
