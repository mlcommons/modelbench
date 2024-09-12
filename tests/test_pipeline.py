from modelgauge.pipeline import Pipeline, Source, Pipe, Sink


class MySource(Source):
    def new_item_iterable(self):
        return [1, 2, 3]


class MyPipe(Pipe):
    def handle_item(self, item):
        return item * 2


class MySink(Sink):
    def __init__(self):
        super().__init__()
        self.results = []

    def handle_item(self, item):
        print(item)
        self.results.append(item)


def test_pipeline_basics():
    p = Pipeline(MySource(), MyPipe(), MySink(), debug=True)
    p.run()
    assert p.sink.results == [2, 4, 6]


class MyExpandingPipe(Pipe):
    def handle_item(self, item):
        self.downstream_put(item * 2)
        self.downstream_put(item * 3)


def test_pipeline_with_stage_that_adds_elements():
    p = Pipeline(
        MySource(),
        MyExpandingPipe(),
        MySink(),
    )
    p.run()
    assert p.sink.results == [2, 3, 4, 6, 6, 9]


def test_source_exception_handling():
    class ExplodingSource(Source):
        def new_item_iterable(self):
            for i in [1, 2, 3]:
                if i % 2 == 1:
                    yield i
                else:
                    raise ValueError()

    p = Pipeline(ExplodingSource(), MyPipe(), MySink(), debug=True)
    p.run()
    assert p.sink.results == [2]  # generator function ends at first exception


def test_pipe_exception_handling():
    class ExplodingPipe(Pipe):
        def handle_item(self, item):
            if item % 2 == 1:
                return item * 2
            raise ValueError("this should get caught")

    p = Pipeline(MySource(), ExplodingPipe(), MySink(), debug=True)
    p.run()
    assert p.sink.results == [2, 6]


# more rich tests are in test_prompt_pipeline
