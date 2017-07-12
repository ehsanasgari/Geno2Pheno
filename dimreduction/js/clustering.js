var canvas = document.getElementById('canvas');
var rect;

var startX = 0;
var startY = 0;

var currentX = 0;
var currentY = 0;
var pushed = false;

var all_points = new Array();

var results = new Array();
var colors = [ "#99008a", "#0008f0", "#f00000", "#f0cc00", "#039900",
		"#00f0c8", "#990000", "#f000d8", "#d0f000", "#04f000", "#008799",
		"#000599", "#99005e", "#992b00", "#859900", "#00f040", "#00d4f0",
		"#520099", "#f00094", "#f04400", "#599900", "#009929", "#0090f0",
		"#8000f0", "#990033", "#995700", "#8cf000", "#009954", "#005c99",
		"#7d0099", "#f00050", "#f08800", "#2e9900", "#00f084", "#004cf0",
		"#c400f0", "#998200", "#48f000", "#009980", "#003099", /* second*/ "#c33", "#00F",
		"#b38686", "#8c3123", "#4c2b26", "#ffa280", "#ff6600", "#ccad99",
		"#bf6600", "#734d00", "#332200", "#ffcc00", "#ccb866", "#555916",
		"#d9ff40", "#eaffbf", "#1b3300", "#3f8c23", "#4d664d", "#00ff66",
		"#66cc9c", "#2daab3", "#13494d", "#8fb6bf", "#003c59", "#3db6f2",
		"#0081f2", "#303840", "#597db3", "#000c59", "#3347cc", "#0000f2",
		"#0000b3", "#333366", "#bfbfff", "#ca79f2", "#c200f2", "#770080",
		"#330030", "#663363", "#cc99c2", "#ff40bf", "#e5005c", "#733949",
		"#e5001f", "#330007" ];

function ClusteringPoints(id, x, y, type, context,time,intensity) {
	this.id = id;
	this.x = x;
	this.y = y;
	this.context = context;
	this.type = type;
	this.selected = false;
	this.time=time;
	this.intensity=intensity;
}


function select_area_point(index) {
	if ($('#select_area_' + index).is(':checked')) {
		$('#select_search_' + index).prop('checked', true);
		all_points[index].selected = true;

		if ($('#selected_label_showing').is(':checked')) {
			textAtCoord(all_points[index]);
		}
	} else {
		$('#select_search_' + index).prop('checked', false);
		all_points[index].selected = false;
		$('canvas').removeLayer('point_' + index).drawLayers();
	}

}

function select_search_point(index) {
	if ($('#select_search_' + index).is(':checked')) {
		$('#select_area_' + index).prop('checked', true);
		all_points[index].selected = true;

		if ($('#selected_label_showing').is(':checked')) {
			textAtCoord(all_points[index]);

		}

	} else {
		$('#select_area_' + index).prop('checked', false);
		all_points[index].selected = false;
		$('canvas').removeLayer('point_' + index).drawLayers();
	}

}

function ClusteringPoint2DIVarea(index, point) {
	var context_words = point.context.replace("\\' \"", "");
	return "<table><tr><td><input type='checkbox' id='select_area_" + point.id
			+ "' " + ((point.selected == true) ? "checked='true'" : "")
			+ " onchange='select_area_point(" + point.id
			+ ")'></td><td width='90%'><div class='point_div' style='color:"
			+ colors[point.type]
			+ ";font-weight: bold;' onmousedown='showCoord(" + point.x + ","
			+ point.y + ");'>" + index + ") " + context_words
			+ "</div></td></tr></table>";

}

function ClusteringPoint2DIVsearch(index, point) {
	var context_words = point.context.replace("\\' \"", "");
	return "<table><tr><td><input type='checkbox' id='select_search_"
			+ point.id + "' "
			+ ((point.selected == true) ? "checked='true'" : "")
			+ " onchange='select_search_point(" + point.id
			+ ")'></td><td width='90%'><div class='point_div' style='color:"
			+ colors[point.type]
			+ ";font-weight: bold;' onmousedown='showCoord(" + point.x + ","
			+ point.y + ");'>" + index + ") " + context_words
			+ "</div></td></tr></table>";

}

function XYPoint(x, y) {
	this.x = x;
	this.y = y;
}

function xy2TSNEXY(point) {
	return new XYPoint((point.x / 2) - 150, 150 - (point.y / 2));
}

function findAllSentences(point1, point2) {
	p1 = xy2TSNEXY(point1);
	p2 = xy2TSNEXY(point2);
	$.each(all_points, function(index, value) {

		if (value.x >= p1.x && value.x <= p2.x && value.y <= p1.y
				&& value.y >= p2.y)
			results.push(value);

	});
	var htmlCode = "";

	$.each(results, function(index, value) {

		htmlCode = htmlCode + ClusteringPoint2DIVarea(index + 1, value);

	});

	$("#dialog").html(htmlCode);
	$("#dialog").dialog({
		height : 500
	});
	$("#dialog").dialog({
		closeText : " "
	});

}
function loading() {

	$("#dialog").html("Loading.. <br/><img src='images/loading.gif'/>");
	$("#dialog").dialog({
		height : 500
	});
	$("#dialog").dialog({
		closeText : " "
	});

}

function searchPoint() {

}

function showClusters() {
	$('#all_label_showing').prop('checked', false);
	$('#selected_label_showing').prop('checked', false);
	$("#clustering_result").css("visibility", "visible");
	var points = $("#coordinate_points").val();
	var points_lines = points.split(/\r|\r\n|\n/);
	var sentences = $("#sentences").val();
	var sentences_lines = sentences.split(/\r|\r\n|\n/);
	
	
	var max = Number.MIN_VALUE;
	var max_time = Number.MIN_VALUE;
	var min_time = Number.MAX_VALUE;
	
	
	$.each(points_lines, function(index, value) {
		coord = value.split('\t');
		max = Math.max(max, Math.abs(coord[0]));
		max = Math.max(max, Math.abs(coord[1]));
		if(coord.length==4)
		{
			max_time = Math.max(max_time, coord[3]);
			min_time = Math.min(min_time, coord[3]);
		}
	});

	$
			.each(
					points_lines,
					function(index, value) {
						coord = value.split('\t');
						coord[1] = (parseFloat(coord[1]) / max) * 130;
						coord[0] = (parseFloat(coord[0]) / max) * 130;
						var yval = 150 - coord[1];
						var xval = coord[0] + 150;

						var id = all_points.length;
						all_points.push(new ClusteringPoints(id, coord[0],
								coord[1], coord[2], sentences_lines[index],(coord.length==4)?coord[3]:1,(coord.length==4)?(Math.pow((coord[3]-min_time)/(max_time-min_time),3)):1));
						$("canvas")
								.drawArc(
										{
											layer : true,
											group : coord[2],
											fillStyle : hexToRgba(colors[coord[2]],(coord.length==4)?(Math.pow((coord[3]-min_time)/(max_time-min_time),3)):1),
											strokeStyle: colors[coord[2]],
											x : xval * 2,
											y : yval * 2,
											radius : 3,
											mouseover : function(layer) {

												$("#sentence")
														.html(
																"<div  style='font-size:small;stext-align: center; color: "
																		+ layer.strokeStyle
																		+ "'><b>"
																		+ sentences_lines[index]
																				.replace(
																						"'",
																						"")
																				.replace(
																						"\"",
																						"")
																		+ "</b></div>");

											},
											mouseout : function(layer) {

												$("#sentence")
														.html(
																"<div style='color: #999;text-align:center'>Select an area to see the contexts</div>");

											}
										});

					});

	$("#coordinates").remove();

	canvas = document.getElementById('canvas');

	$("#canvas").css("cursor", "pointer");

	$("#canvas").on(

	'mousedown',

	function(event) {

		$("#canvas").css("cursor", "crosshair");

		rect = canvas.getBoundingClientRect();

		startX = event.clientX - rect.left;

		startY = event.clientY - rect.top;

		pushed = true;

		$('canvas').drawImage({
			layer : true,
			name : 'crosshair',
			source : 'images/crosshairs.png',
			x : startX - 10,
			y : startY - 10,
			fromCenter : false
		});

		$("#canvas").on(

		'mousemove',

		function(event) {

			if (pushed == true) {

				x1 = Math.min(startX, event.clientX - rect.left);
				x2 = Math.max(startX, event.clientX - rect.left);
				y1 = Math.min(startY, event.clientY - rect.top);
				y2 = Math.max(startY, event.clientY - rect.top);

				a = new XYPoint(x1, y1);

				b = new XYPoint(x2, y2);

				var ctx = canvas.getContext("2d");

				ctx.beginPath();
				ctx.lineWidth = "2";
				ctx.strokeStyle = "rgba(250,170,22,0.3)";
				ctx.rect(x1, y1, x2 - x1, y2 - y1);
				ctx.stroke();

				loading();
				// position = $(this).position();
				$("#dialog").dialog('option', 'position', [ 80, 150 ]);

				$("#canvas").on(

				'mouseup',

				function(event) {

					if (pushed == true) {
						results = new Array();
						findAllSentences(a, b);
						$("#dialog").dialog();
						$("#canvas").css("cursor", "pointer");
						$('canvas').removeLayer('crosshair').drawLayers();
						pushed = false;
					}
				});
			}

		});

	});
	showSearch();
	$("#filter").val('');
}

function showSearch() {

	$("#filter").keyup(
			function() {
				var htmlCode = "";
				$("#search_res").html("");
				// Retrieve the input field text and reset the count to zero
				var filter = $(this).val(), count = 0;

				// Loop through the comment list
				$.each(all_points, function(index, value) {

					// If the list item does not contain the text phrase fade it
					// out
					if (value.context.search(new RegExp(filter, "i")) >= 0) {
						htmlCode = htmlCode
								+ ClusteringPoint2DIVsearch(count + 1, value);
						count++;
					}
				});
				$("#search_res").html(htmlCode);
				// Update the count
				$("#filter-count").text("contexts = " + count);
			});

}

function handleAllLabels() {
	$('canvas').removeLayerGroup('labels').drawLayers();

	if ($('#all_label_showing').is(':checked')) {
		$('#selected_label_showing').prop('checked', false);

		$.each(all_points, function(index, value) {
			textAtCoord(value);
		});
	}

}

function handleSelectedLabels() {
	$('canvas').removeLayerGroup('labels').drawLayers();

	if ($('#selected_label_showing').is(':checked')) {
		$('#all_label_showing').prop('checked', false);
		$.each(all_points, function(index, value) {

			if (value.selected == true) {
				textAtCoord(value);
			}
		});

		$("#filter").keyup();

	}
}

function textAtCoord(point) {

	var yval = 150 - parseFloat(point.y);
	var xval = parseFloat(point.x) + 150;
	$('canvas').drawText({
		layer : true,
		group : 'labels',
		name : 'point_' + point.id,
		fillStyle : colors[point.type],
		x : xval * 2 - 35-point.context.length,
		y : yval * 2 - 15,
		fromCenter : false,
		text : point.context,
		fontSize : 10,
		fontFamily : 'Verdana, sans-serif',
	});

}

function showCoord(x, y) {

	var yval = 150 - parseFloat(y);
	var xval = parseFloat(x) + 150;
	$('canvas').drawImage({
		source : 'images/map-point.png',
		x : xval * 2 - 15,
		y : yval * 2 - 40,
		fromCenter : false
	});

}

function hexToRgba(str, intensity) {
	if (/^#([0-9a-f]{3}|[0-9a-f]{6})$/ig.test(str)) {
		var hex = str.substr(1);
		hex = hex.length == 3 ? hex.replace(/(.)/g, '$1$1') : hex;
		var rgb = parseInt(hex, 16);
		return 'rgba('
				+ [ (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255 ].join(',')
				+ ',' + intensity + ')';
	}

	return false;
}
