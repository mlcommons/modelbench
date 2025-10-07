import click

from modelgauge.annotator_registry import ANNOTATOR_MODULE_MAP, ANNOTATORS
from modelgauge.load_namespaces import load_namespaces, load_namespace
from modelgauge.sut_definition import SUTDefinition
from modelgauge.sut_factory import LEGACY_SUT_MODULE_MAP, SUT_FACTORY

LOAD_ALL = "__load_all_namespaces__"


class LazyModuleImportGroup(click.Group):
    """Modified from https://click.palletsprojects.com/en/stable/complex/#defining-the-lazy-group"""

    def __init__(self, *args, lazy_lists=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_lists = lazy_lists or {}

    def resolve_command(self, ctx, args):
        cmd_name, cmd, args = super().resolve_command(ctx, args)
        if cmd_name in self.lazy_lists:
            self._lazy_load(cmd_name)

        # now we lazy load any additional modules based on the command line args
        # we have to copy args as make_context mutates it
        cmd_ctx = cmd.make_context(cmd_name, args.copy(), parent=ctx, resilient_parsing=True)

        test_name = cmd_ctx.params.get("test")
        if test_name:
            load_namespace("tests")

        # we use both sut and sut_uid
        maybe_sut_uid = cmd_ctx.params.get("sut") or cmd_ctx.params.get("sut_uid")

        if maybe_sut_uid:
            # resolve the sut uid in case a sut definition is provided
            sut_uid = SUTDefinition.canonicalize(maybe_sut_uid)
            if not SUT_FACTORY.knows(sut_uid):
                if sut_uid not in LEGACY_SUT_MODULE_MAP:
                    raise ValueError(
                        f"Unknown SUT '{sut_uid}' and no legacy mapping found. Did you forget to add it to sut_factory.LEGACY_SUT_MODULE_MAP?"
                    )
                load_namespace(f"suts.{LEGACY_SUT_MODULE_MAP[sut_uid]}")

        annotator_uids = cmd_ctx.params.get("annotator_uids")
        if annotator_uids:
            # try loading the private annotators
            try:
                import modelgauge.private_ensemble_annotator_set
            except NotImplementedError:
                pass
            for annotator_uid in annotator_uids:
                if not ANNOTATORS.knows(annotator_uid):
                    if annotator_uid not in ANNOTATOR_MODULE_MAP:
                        raise ValueError(f"Unknown annotator '{annotator_uid}' and no mapping found.")
                    load_namespace(f"annotators.{ANNOTATOR_MODULE_MAP[annotator_uid]}")

        return cmd_name, cmd, args

    def _lazy_load(self, cmd_name):
        namespaces_to_load = self.lazy_lists[cmd_name]
        if namespaces_to_load == LOAD_ALL:
            load_namespaces()
        else:
            load_namespace(namespaces_to_load)
