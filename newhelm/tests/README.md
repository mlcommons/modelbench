# Test plugins

NewHELM uses [namespace plugins](../../docs/plugins.md) to separate the core libraries from the implementations of specific Tests. That way you only have to install the dependencies you actually care about.

Any file put in this directory, or in any installed package with a namespace of `newhelm.tests`, will be automatically loaded by the NewHELM command line tool via `load_plugins()`.
