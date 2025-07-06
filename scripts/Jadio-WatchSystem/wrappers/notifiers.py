def send_notification(message, title=None, provider='email', **kwargs):
    try:
        from notifiers import get_notifier
        notifier = get_notifier(provider)
        params = {'message': message}
        if title:
            params['title'] = title
        params.update(kwargs)
        return notifier.notify(**params)
    except ImportError:
        raise RuntimeError('notifiers package is not installed')
    except Exception as e:
        return {'error': str(e)}
