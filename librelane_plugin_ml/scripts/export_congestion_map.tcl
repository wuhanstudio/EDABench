source $::env(SCRIPTS_DIR)/openroad/common/io.tcl

read_current_odb

gui::show {
gui::pause 2000
gui::dump_heatmap Pin pin.map
gui::dump_heatmap Placement placement.map
gui::dump_heatmap Routing routing.map
gui::dump_heatmap RUDY rudy.map
} false
