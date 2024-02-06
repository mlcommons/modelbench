# Plugins

NewHELM is designed to be extensible using [namespace package plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages). This allows you to write code which interacts with the core of NewHELM which can live only in your local file system or in your own separate package.

## Basics of plugins

To discover plugins, NewHELM searches special namespaces within `newhelm`. A full list of supported namespaces and the code for how this works can be found in [load_plugins.py](/newhelm/load_plugins.py). For example, any module in `newhelm.tests` will be imported as part of the `load_plugins()` call. This makes any code in that module accessible via reflection (e.g. finding all subclasses of a base class), as well as via instance factories (PR #20).

## Adding a plugin to a local checkout of NewHELM

If you have a local checkout of NewHELM, you can add your module by creating a new file in the desired namespace. For example, if you want to add a Test, you can create a new file in `newhelm/tests/`.

### Adding a plugin to NewHELM's repository

If you would like to create a plugin that is distributed as part of the NewHELM repository, there are a few points of guidance. First, if you are adding a Test that doesn't require any additional poetry dependencies, you can put your files in `plugins/standard_tests/newhelm/tests/`, and skip the rest of this section.

TODO: Write the guidance for adding a plugin requiring a dependency.

## In your own package

NewHELM also supports distributing your plugin in its own package. Lets assume you want to call it `mycoolplugins`. Using this guide, if someone wanted to use your plugins, they could do so with the following commands:

```
pip install newhelm
pip install mycoolplugins
```

Now any runs of NewHELM will automatically discover every plugin you wrote in `mycoolplugins`. Furthermore, other plugin writers can import your plugins just like they were written in core NewHELM. To make this magic work:

1. In your package, recreate the `newhelm/<namespace>` directory structure.
1. Do **NOT** create any `__init__.py` files in those directories. The absence of those files tells python these are namespaces.
1. Add your files in the desired directory (e.g. `newhelm/tests/super_cool_test.py`)
1. You can now distribute your package however you want.

For an example of how this works, see the [demo_plugin](https://github.com/mlcommons/newhelm/tree/main/demo_plugin) directory.
