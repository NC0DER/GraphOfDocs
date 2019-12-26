//Viz is a global object and it is created once.
var viz;

$(document).ready(function () {
    var query = "MATCH (n:Word)-[r:connects]-(k) "
        + "WHERE n.pagerank > 5 "
        + "AND k.pagerank > 5 "
        + "AND n.pagerank < 10 "
        + "AND k.pagerank < 10 "
        + "RETURN n,r,k LIMIT 1000";
    draw(query);
});

$("#query").click(function () {
    var start = $("#field1").val();
    var end = $("#field2").val();
    if (start === "" || end === "") {
        alert("Please speficy the ranges!");
        return;
    }
    // Build the query based on the above values.
    var query = "MATCH (n:Word)-[r:connects]-(k) "
        + "WHERE n.pagerank > " + start + " "
        + "AND k.pagerank > " + start + " "
        + "AND n.pagerank < " + end + " "
        + "AND k.pagerank < " + end + " "
        + "RETURN n,r,k LIMIT 1000";
    viz.renderWithCypher(query);
});

$("#stabilize").click(function () {
    viz.stabilize();
})

$("#textarea").keyup(function (e) {
    var code = e.keyCode ? e.keyCode : e.which;
    if (code === 13) { // Enter key pressed.
        var query = $("#textarea").val();
        if (query === ""){
            alert("Please supply a query!");
            return;
        }
        viz.renderWithCypher(query);
        return;
    }
});

function draw(query) {
    // Create a config object for viz.
    var config = {
        container_id: "viz",
        server_url: "bolt://localhost:7687",
        server_user: "neo4j",
        server_password: "123",
        labels: {
            "Word": {
                caption: "key",
                size: "pagerank",
                community: "community"
            }
        },
        relationships: {
            "connects": {
                caption: "weight",
                thickness: "count"
            }
        },
        initial_cypher: query
    }
    viz = new NeoVis.default(config);
    viz.render();
    return 
}
