"""Install header-driven BTCPFX01 reading in a recognized offline replay checkout."""
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("native reader interface changed: " + old[:70])
    return text.replace(old, new, 1)


def install(entry):
    source = entry.read_text()
    model = entry.with_name("MetadataReplay.cc")
    if "--btcpfx-reader-version" in source:
        return
    source = replace_once(source, '    try {\n        stateRequire(', '''    try {
        if (argc == 2 && std::string(argv[1]) == "--btcpfx-reader-version") {
            std::cout << "2" << std::endl;
            return 0;
        }
        stateRequire(''')
    source = replace_once(source, 'config(const std::string& value)', 'config(const std::string& value, uint32_t base)')
    source = replace_once(source, 'c.block_tokens      = 256;', 'c.block_tokens      = base;')
    source = replace_once(source, 'c.swa_block_tokens  = 1024;', 'c.swa_block_tokens  = c.block_tokens * c.cp_size;')
    anchor = '        const auto identity   = contents(options.at("--identity"));'
    source = replace_once(source, anchor, anchor + '''
        std::ifstream first(input / "000000.btcbin", std::ios::binary);
        char first_magic[8];
        first.read(first_magic, sizeof(first_magic));
        stateRequire(first && std::memcmp(first_magic, "BTCPFX01", 8) == 0, "invalid trace header");
        const auto base = read<uint32_t>(first);
        stateRequire(base > 0 && base <= INT32_MAX / 4, "invalid CP4 block granularity");''')
    source = replace_once(source, 'config(parameters)', 'config(parameters, base)')
    source = replace_once(source, 'index = ordinal / 100000, skip = ordinal % 100000;', 'index = 0, skip = ordinal;')
    source = replace_once(source, 'read<uint32_t>(f) == 256', 'read<uint32_t>(f) == base')
    source = replace_once(source, 'count > 0 && count <= 100000', 'count > 0')
    anchor = '            uint64_t chunk_tokens = 0;'
    source = replace_once(source, anchor, '''            if (skip >= count) {
                skip -= count;
                ++index;
                continue;
            }
            uint64_t rows_read = 0;
''' + anchor)
    source = replace_once(source, '                (void)text(f);', '''                ++rows_read;
                const auto id_length = read<uint32_t>(f);
                stateRequire(id_length <= 1048576, "invalid request id length");
                f.ignore(id_length);''')
    a = source.index('                q.prefix_keys.resize(q.input_tokens / 256);')
    b = source.index('                stateRequire(q.timestamp_ns', a)
    source = source[:a]+'''                const auto keys = static_cast<size_t>(q.input_tokens) / base;
                chunk_tokens += q.input_tokens;
                if (i < skip) {
                    f.ignore(keys * sizeof(int64_t));
                    stateRequire(bool(f), "truncated skipped request");
                    continue;
                }
                q.prefix_keys.resize(keys);
                f.read(reinterpret_cast<char*>(q.prefix_keys.data()), keys * sizeof(int64_t));
                stateRequire(bool(f), "truncated trace keys");
'''+source[b:]
    source = replace_once(source, 'if (ordinal % 100000 == 0)', 'if (rows_read == count)')
    model_source = replace_once(model.read_text(), 'o.swa && o.block_tokens == 256 && o.cp_size == 4',
                                'o.swa && o.block_tokens > 0 && o.cp_size == 4')
    model_source = model_source.replace('CP4, base/kernel 256', 'CP4, trace base/kernel granularity')
    # Warm-up still performs every cache action and invariant check; its result
    # never enters EvaluationStats. Avoid allocating timing/eviction map entries.
    import re
    model_source, changed = re.subn(r'(?m)^( +)(out\.stage_us\[.*?;|out\.evictions\s*= eviction_counts;)$',
                                    r'\1if (r.observe)\n\1    \2', model_source)
    if changed == 0:
        raise ValueError("unknown warm-up observation boundary")
    model.write_text(model_source)
    entry.write_text(source)
