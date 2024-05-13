# Plugins

ModelGauge is designed to be extensible using [namespace package plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages). This allows you to write code which interacts with the core of ModelGauge which can live only in your local file system or in your own separate package.

## Basics of plugins

To discover plugins, ModelGauge searches special namespaces within `modelgauge`. A full list of supported namespaces and the code for how this works can be found in [load_plugins.py](/modelgauge/load_plugins.py). For example:

* Files defining a Test should go in the `modelgauge.tests` namespace.
* Files defining a SUT should go in the `modelgauge.suts` namespace.

The `load_plugins()` call will import all modules in all namespace directories. This makes any code in that module accessible via reflection (e.g. finding all subclasses of a base class) and will run [InstanceFactory.register](https://github.com/mlcommons/modelgauge/blob/main/modelgauge/instance_factory.py) calls. This lets the ModelGauge command line list all Tests/SUTs without having to edit any core library code.

## Adding a plugin to a local checkout of ModelGauge

If you have a local checkout of ModelGauge, you can add your module by creating a new file in the desired namespace. For example, if you want to add a Test, you can create a new file in `modelgauge/tests/`.

### Adding a plugin to ModelGauge's repository

If you would like to create a plugin that is distributed as part of the ModelGauge repository, there are a few points of guidance. First, if you are adding a Test that doesn't require any additional poetry dependencies, you can put your files in `plugins/standard_tests/modelgauge/tests/`, and skip the rest of this section.

TODO: Write the guidance for adding a plugin requiring a dependency.

## From a local directory
It is possible to load plugins from a local directory for certain commands using the CLI option `--plugin-dir`. For example:

```shell
modelgauge run-sut --sut mycoolplugin --plugin-dir /my/plugins --prompt "Can you answer this question?"
```

> [!WARNING]
> `--plugin-dir` will import any modules in the specified directory which can execute code that could be harmful, malicious, 
> or that could have unexpected consequences. Use with caution and in a trusted environment.

## In your own package

ModelGauge also supports distributing your plugin in its own package. Lets assume you want to call it `mycoolplugin`. Using this guide, if someone wanted to use your plugins, they could do so with the following commands:

```
pip install modelgauge
pip install mycoolplugin
```

Now any runs of ModelGauge will automatically discover every module you wrote in `mycoolplugin`. Furthermore, other plugin writers can import your plugin just like they were written in core ModelGauge. To make this magic work:

1. In your package, recreate the `modelgauge/<namespace>` directory structure.
1. Do **NOT** create any `__init__.py` files in those directories. The absence of those files tells python these are namespaces.
1. Add your files in the desired directory (e.g. `modelgauge/tests/super_cool_test.py`)
1. You can now distribute your package however you want.

For an example of how this works, see the [demo_plugin](https://github.com/mlcommons/modelgauge/tree/main/demo_plugin) directory.
