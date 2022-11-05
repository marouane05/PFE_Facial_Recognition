
// Onclick of the button
//document.getElementById("openR").onclick = function () {
  // Call python's random_python function
  /*eel.random_python()(function(number){
    // Update the div with a random number returned by python
    document.querySelector(".random_number").innerHTML = number;
  })*/

 /*
  eel.openFaceRecognition();
}*/
document.querySelector("button").onclick = function () {
  // Call python's random_python function
 eel.openFaceRecognition();
}
/*
document.getElementById("btnnrrrs").onclick=function(){
  // Call python's random_python function
 eel.openFaceRecognition();
} */
function ToAddP(){
//console.log("click");

}
function next3(){
 var nom = document.getElementById("nom").value;
 var infos = document.getElementById("role").value;
 eel.TakePictures(nom,infos);
}
function openSystem(){
eel.OpenSystem()
}


        // When the user scrolls down 20px from the top of the document, show the button
        window.onscroll = function () {
            scrollFunction()
        };

        function scrollFunction() {
            if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                document.getElementById("movetop").style.display = "block";
            } else {
                document.getElementById("movetop").style.display = "none";
            }
        }

        // When the user clicks on the button, scroll to the top of the document
        function topFunction() {
            document.body.scrollTop = 0;
            document.documentElement.scrollTop = 0;
        }
    <!-- //move top -->

    /*common jquery plugin
    <script src="assets/js/jquery-3.3.1.min.js"></script>-->
    <script src="https://code.jquery.com/jquery-3.3.1.min.js"
    integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8=" crossorigin="anonymous"></script>
    */

     //common jquery plugin



    // MENU-JS

        $(window).on("scroll", function () {
            var scroll = $(window).scrollTop();

            if (scroll >= 80) {
                $("#site-header").addClass("nav-fixed");
            } else {
                $("#site-header").removeClass("nav-fixed");
            }
        });

        //Main navigation Active Class Add Remove
        $(".navbar-toggler").on("click", function () {
            $("header").toggleClass("active");
        });
        $(document).on("ready", function () {
            if ($(window).width() > 991) {
                $("header").removeClass("active");
            }
            $(window).on("resize", function () {
                if ($(window).width() > 991) {
                    $("header").removeClass("active");
                }
            });


        });

  //MENU-JS

  //disable body scroll which navbar is in active

        $(function () {
            $('.navbar-toggler').click(function () {
                $('body').toggleClass('noscroll');
            })
        });


(function(){
	if(typeof _bsa !== 'undefined' && _bsa) {
  		// format, zoneKey, segment:value, options
  		_bsa.init('flexbar', 'CKYI627U', 'placement:w3layoutscom');
  	}
})();

(function(){
if(typeof _bsa !== 'undefined' && _bsa) {
	// format, zoneKey, segment:value, options
	_bsa.init('fancybar', 'CKYDL2JN', 'placement:demo');
}
})();

(function(){
	if(typeof _bsa !== 'undefined' && _bsa) {
  		// format, zoneKey, segment:value, options
  		_bsa.init('stickybox', 'CKYI653J', 'placement:w3layoutscom');
  	}
})();


  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'UA-149859901-1');

     window.ga=window.ga||function(){(ga.q=ga.q||[]).push(arguments)};ga.l=+new Date;
     ga('create', 'UA-149859901-1', 'demo.w3layouts.com');
     ga('require', 'eventTracker');
     ga('require', 'outboundLinkTracker');
     ga('require', 'urlChangeTracker');
     ga('send', 'pageview');

    let current_url = document.location;
    document.querySelectorAll(".navbar-nav .nav-link").forEach(function(e){
       if(e.href == current_url){
         // e.classList += " current";
        //  document.querySelectorAll(".navbar-nav .nav-link")[0].parentElement.classList.add("active")
          e.parentElement.classList.add("active");
       }
    });



function ouvrir(identifier){
    var ids = $(identifier).data('id');
    console.log(""+ids);
/*var e = document.getElementById("p"+ids).textContent;
        console.log("voilà le text"+e);
  */


  $(".modal-title").html(""+document.getElementById("t"+ids).textContent);
    $(".modal-body #text").html(""+document.getElementById("p"+ids).value);

    }


$("#sabonner").click(function(e){
    e.preventDefault();
    console.log("hi");
    var email = $("#emaila").val();
    $.ajax({
       type: 'POST',
       url: 'abonner.php',
       data: {
           email : email
       },
       success: function(response){
        console.log(response);
        if(response.error==true){
       alert("oops! erreur");
   } else {
       //$('#contactp').html("<h1>Merci votre message a bien été enregistrée</h1>");
    //   alert("good");
       $("#subs").html("<h3 class='text-white mt-1' >Merci pour votre abonnement !</h3>")
    }
       }
    })

})







