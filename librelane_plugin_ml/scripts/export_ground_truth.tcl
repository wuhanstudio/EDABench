source $::env(SCRIPTS_DIR)/openroad/common/io.tcl

read_current_odb

gui::show {
gui::pause 2000
gui::dump_heatmap Routing routing_gt.map
} false
